import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import ops.geometry as geo

from torchsparse.tensor import PointTensor
from loguru import logger
from utils import apply_log_transform, vis_2D_depth_prediction_comparison
from src.modules import SPVCNN
from src.gru_fusion import GRUFusion
from ops.back_project import back_project
from ops.generate_grids import generate_grid
from ops.geometry import batched_build_img_pts_tensor
from torch_scatter import scatter
from src.semseg_heads import SemSegHead3D
from src.semseg_cdr_link import SemsegCDRLink
from src.depth_refinement import HypothesisDecoder, AnchorFeatureDecoder
from src.depth_upsample_propagator import PropagationNet


class CoarseToFineFragNet(nn.Module):
    """
    Coarse-to-fine network, where both anchor occupancy refinement and point-to-vertex matching refinement are done
    on top of the metric-semantic GRU fusion.
    Input to this network needed to be n_view fragments (including the depth batch).
    """

    def __init__(self, cfg):
        super(CoarseToFineFragNet, self).__init__()
        self.cfg = cfg
        alpha = int(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        ch_in = [80 * alpha + 1,
                 96 + 40 * alpha + 2 + 1,
                 48 + 24 * alpha + 2 + 1,
                 24 + 24 + 2 + 1]
        gru_input_channels = [96, 48, 24]
        self.backbone2d_output_channels = [80, 40, 24]  # c/m/f

        if self.cfg.FUSION.FUSION_ON:
            # GRU Fusion
            self.gru_fusion = GRUFusion(cfg, gru_input_channels)

        self.sp_convs = nn.ModuleList()
        self.tsdf_preds = nn.ModuleList()
        self.occ_preds = nn.ModuleList()

        for i in range(len(cfg.THRESHOLDS)):
            self.sp_convs.append(
                SPVCNN(in_channels=ch_in[i],
                       pres=1,
                       cr=1 / 2 ** i,
                       vres=self.cfg.VOXEL_SIZE * 2 ** (self.cfg.N_STAGE - 1 - i),
                       dropout=self.cfg.SPARSEREG.DROPOUT)
            )
            self.tsdf_preds.append(nn.Linear(gru_input_channels[i], 1))
            self.occ_preds.append(nn.Linear(gru_input_channels[i], 1))
        self.semseg_pred = SemSegHead3D(cfg, in_channels=gru_input_channels)

        if self.cfg.DEPTH_PREDICTION:
            if self.cfg.CDR.DEPTH_PRED:
                self.decoder = HypothesisDecoder(in_dim=gru_input_channels[0] + self.backbone2d_output_channels[0],
                                                 h_dim=128, kernel_size=3, padding=1)
                self.depth_propagator = nn.ModuleList()
                for i in range(len(cfg.THRESHOLDS)):
                    self.depth_propagator.append(PropagationNet(in_dim=self.backbone2d_output_channels[i] + 1))
                self.ref_idx = torch.tensor([2, 3, 4, 5, 6], dtype=torch.int64)

            if self.cfg.CDR.SEMSEG_REFMNT:  # semseg refmnt
                self.img_size_for_linking = (640, 480)
                self.linkers = nn.ModuleList()
                self.easy_mink_decoder = nn.ModuleList()
                self.sparse_interpolation = ME.MinkowskiInterpolation()
                from MinkowskiEngine.modules.resnet_block import BasicBlock
                for cb2d, c3d in zip(self.backbone2d_output_channels, gru_input_channels):
                    self.linkers.append(SemsegCDRLink(cb2d, c3d,
                                                      view_num=9))  # since n_view == 9 for each frag
                    self.easy_mink_decoder.append(BasicBlock(inplanes=c3d, planes=c3d,
                                                             dimension=3))  # with back projection, 3d feat is D=2, (#voxel, c+1)
            if self.cfg.CDR.FEAT_REFMNT:
                self.anchor_feature_decoder = nn.ModuleList()  # anchor occupancy refmnt: sparse_encodec_convs
                for i in range(len(cfg.THRESHOLDS)):
                    self.anchor_feature_decoder.append(
                        AnchorFeatureDecoder(
                            cfg,
                            in_dim=gru_input_channels[2 - i],  # in scale sequence
                            out_dim=gru_input_channels[2 - i],
                            n_vox=gru_input_channels[i]))  # deprecated comment: at coarse: x_var (5, 80, 1200)

    @staticmethod
    def upsample(pre_feat, pre_coords, interval,
                 depth_pred_dict, idx_of_stage,
                 stage_name_list, feats_2d, num=8,
                 depth_enabled=False):
        """ nearest neighbour interpolation
        @param pre_feat: (Tensor), features from last level, (N, C)
        @param pre_coords: (Tensor), coordinates from last level, (N, 4) (4 : Batch ind, x, y, z)
        @param interval: interval of voxels, interval = 2 ** scale
        @param num: 1 -> 8

        @return: up_feat : (Tensor), upsampled features, (N*8, C)
        @return: up_coords: (N*8, 4), upsampled coordinates, (4 : Batch ind, x, y, z)
        """
        with torch.no_grad():
            pos_list = [1, 2, 3, [1, 2], [1, 3], [2, 3], [1, 2, 3]]  # x,y,z,xy,xz,yz,xyz
            n, c = pre_feat.shape
            up_feat = pre_feat.unsqueeze(1).expand(-1, num,
                                                   -1).contiguous()  # .continuous() expands overwrite pre_feat mem and rename as up_feat
            up_coords = pre_coords.unsqueeze(1).repeat(1, num, 1).contiguous()
            for i in range(num - 1):
                # leave the (:,0,:) unchanged, and then nearest interpolate for (:,1:7,:) according to pos_list
                up_coords[:, i + 1, pos_list[i]] += interval

            up_feat = up_feat.view(-1, c)
            up_coords = up_coords.view(-1, 4)

            if depth_enabled:
                # this upsample func always used when idx_of_stage > 0
                depth_pred_dict[f'depth_{stage_name_list[idx_of_stage]}'] = \
                    F.interpolate(depth_pred_dict[f'depth_{stage_name_list[idx_of_stage - 1]}'].unsqueeze(1),
                                  feats_2d[2 - idx_of_stage].shape[-2:], mode='nearest').squeeze(1)

        return up_feat, up_coords, depth_pred_dict

    @staticmethod
    def get_target(coords, inputs, scale):
        """ won't be used when 'fusion_on' flag is turned on
        @param coords: (Tensor), coordinates of voxels, (N, 4) (4 : Batch ind, x, y, z)
        @param inputs: (List), inputs['tsdf_list' / 'occ_list']: ground truth volume list, [(B, DIM_X, DIM_Y, DIM_Z)]
        @param scale:
        @return tsdf_target: (Tensor), tsdf ground truth for each predicted voxels, (N,)
        @return occ_target: (Tensor), occupancy ground truth for each predicted voxels, (N,)
        """
        with torch.no_grad():
            tsdf_target = inputs['tsdf_list'][scale]
            occ_target = inputs['occ_list'][scale]
            coords_down = coords.detach().clone().long()
            # 2 ** scale == interval
            coords_down[:, 1:] = torch.div(coords[:, 1:], 2 ** scale, rounding_mode='trunc')
            tsdf_target = tsdf_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            occ_target = occ_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            return tsdf_target, occ_target

    # for semseg ablation
    @staticmethod
    def get_sparse_semseg_targets(coords, inputs, scale):
        """ when fusion is turned on, get semseg target while not rerunning gru_fusion()
        @param coords:
        @param inputs:
        @param scale:
        @return:
        """
        with torch.no_grad():
            semseg_target = inputs['semseg_list'][scale]
            coords_down = coords.detach().clone().long()
            coords_down[:, 1:] = torch.div(coords[:, 1:], 2 ** scale, rounding_mode='trunc')
            semseg_target = semseg_target[
                coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]

        return semseg_target

    def process_sparse_feat_to_dict(self, sparse_geo_feats, sparse_coords, interval):
        """Process the sparse 3D geometric feats generated from GRU fusion module,
         to create sparse feature dict for pointflow.
        * convert 3D geometric feats input from `torchsparse.PointTensor` into ME.SparseTensor
        for hypothesisNet interpolation usage
        @return: dict_s: dict of {
            'feats': None, dense feature, but not required at this point
            'pts': (n, 3), pt coordinates
            'batch': (n,), the batch index for each 3d pt
            'res':
            'stride':
            'sparse': Minkowski Sparse Tensor, shape of (n, c), [pre_feat, pre_coord]
        }
        """
        # torchsparse coord to me coord
        me_coords = torch.index_select(sparse_coords, 1, torch.cuda.LongTensor([3, 0, 1, 2]))

        # since ME only takes int coord, 1e4 should not affect the sparse interpolation in hyp refinement
        me_sparsetensor = ME.SparseTensor(sparse_geo_feats, me_coords.int())
        dict_s = {
            'pts': me_sparsetensor.C[:, 1:],  # [n, 3]
            'res': me_sparsetensor.tensor_stride[0] * self.cfg.VOXEL_SIZE * interval,
            'batch': torch.zeros(me_sparsetensor.F.shape[0], dtype=torch.long, device='cuda'),
            'stride': me_sparsetensor.tensor_stride[0],
            'sparse': me_sparsetensor
        }

        return dict_s

    def depth_offset_and_feat_refmnt_wth_pointflow(self, x_sparse,
                                                   depthmap_current_scale, depth_batch_idx,
                                                   img_feats, scale,
                                                   rotmats, tvecs, K, ref_src_edges, dbatch_img_size, offset, n):
        """Point flow algorithm, distinct from 3dvnet, the point flow algo in CDRNet predicts an offset for sparse 3D features. Takes sparse features volume and depth prediction input. pts_img_xx denotes pts in 2D.
        @param x_sparse: containing sparse features from gru fusion (or other 3D ConvNet such as u-net),
        regarded as the scene-prior that to refine depth in three different resolutions.
            dict of {
                'pts': (n, 3), pt coordinates
                'batch': (n,), the batch index for each 3d pt
                'res':
                'stride':
                'sparse': Minkowski Sparse Tensor, shape of (n, c), [pre_feat, pre_coord]
            }
        @param depthmap_current_scale: depth prediction (or a downsampled one from the level of cross dimension refined net), (n_imgs, h, w)
        @param depth_batch_idx: specify the batch index of each image
        @param img_feats: dense 2d features extracted from backbone 2d on images
        @param scale: from 2 to 0, indicates coarse to fine, matched with tsdf gt, 2 to 0
        *********** params from dbatch
        @param rotmats:
        @param tvecs:
        @param K:
        @param ref_src_edges: ref-src pairs, with [0] as ref indices list, [1] as src indices list
        @param dbatch_img_size:
        @param offset: list of the depth offset resolutions that used in pt flow algo
        @param n: num of hypothesis point for each side (before/behind)
        """
        n_views = depthmap_current_scale.shape[0]
        ref_idx, gather_idx = torch.unique(ref_src_edges[0],
                                           return_inverse=True)  # return unique value ref idx, with new index for each ele in ref_src_edges

        with torch.no_grad():
            # first, back project depth pts to determine the point hypotheses
            K_inv = torch.inverse(K[ref_idx])
            R_T = rotmats[ref_idx].transpose(2, 1)  # orthogonal mat R
            pts_img_grid_homo = batched_build_img_pts_tensor(n_views, dbatch_img_size, depthmap_current_scale.shape[1:])
            pts_img_grid_homo = pts_img_grid_homo.type_as(depthmap_current_scale)  # [n_view, 3, n_pts]
            pts_batch_idx = depth_batch_idx.unsqueeze(1).expand(n_views,
                                                                depthmap_current_scale.shape[1] *
                                                                depthmap_current_scale.shape[2]).reshape(
                -1)  # ind of each pt in depth map

            n_pts = pts_img_grid_homo.shape[2]  # per view

            pts_hyp = torch.empty((n_views, 3, n * 2 + 1, n_pts), dtype=torch.float32,
                                  device=depthmap_current_scale.device)  # [n_view, coord, n_hyp, n_pts]

            # back projection of hypothesis pts on ref frames
            for i in range(-n, n + 1):
                # generate hypothesis pt offset version of the depthmap
                # from the original depthmap current scale of ref frame
                pts_img_wth_offset_homo = pts_img_grid_homo * (depthmap_current_scale.view(n_views, 1, -1) + i * offset)
                # hyp version depthmap wrt the ref frame
                # (i=0 is depth_current_scale itself) back projection to have 3d pts
                pts = torch.bmm(R_T, torch.bmm(K_inv, pts_img_wth_offset_homo) - tvecs[ref_idx].unsqueeze(-1))
                pts_hyp[..., i + n, :] = pts

            # second, re-project all pts (including hyp_pts) into the corresponding src images
            # to calculate the variance feature
            n_hpts = (n * 2 + 1) * n_pts
            w = torch.ones((n_views, 1, n_hpts), dtype=torch.float32).type_as(pts_hyp)
            pts_hyp_homo = torch.cat((pts_hyp.view(n_views, 3, n_hpts), w), dim=1)
            P = torch.cat((rotmats, tvecs[..., None]), dim=2)
            P = torch.bmm(K, P)
            # re-projection of hyp pts on src frames
            pts_img_src_hyp_homo = torch.bmm(P[ref_src_edges[1]], pts_hyp_homo[gather_idx])
            z_buffer = pts_img_src_hyp_homo[:, 2]
            z_buffer = torch.abs(z_buffer) + 1e-8  # prevent div/0
            pts_img_hyp = pts_img_src_hyp_homo[:, :2] / z_buffer[:, None]

            grid = pts_img_hyp.transpose(2, 1).view(pts_img_hyp.shape[0], n_hpts, 1, 2)
            grid[..., 0] = (grid[..., 0] / float(dbatch_img_size[1] - 1)) * 2 - 1.0  # normalize to [-1, 1]
            grid[..., 1] = (grid[..., 1] / float(dbatch_img_size[0] - 1)) * 2 - 1.0

        # irregular sampling the hyp pts reprojection 2d result on src frame features
        x = F.grid_sample(img_feats[ref_src_edges[1]], grid, mode='bilinear', align_corners=True)
        x = x.squeeze(3)
        # multi view feature matching for the input 3D geometric feat (or feat-rich pt cloud), using var aggr as in point-mvsnet
        x_avg = scatter(x, gather_idx, dim=0, reduce='mean')
        x_avg_sq = scatter(x ** 2, gather_idx, dim=0, reduce='mean')
        x_var = x_avg_sq - x_avg ** 2

        # the hyp pts reprojection var can be viewed as feat, which contains the hyp pts choice to be determined
        pts_feat = x_var.view(n_views, self.backbone2d_output_channels[0], 2 * n + 1, n_pts) \
            .permute(0, 3, 2, 1) \
            .reshape(n_pts * n_views, 2 * n + 1, self.backbone2d_output_channels[0])
        pts_hyp = pts_hyp.permute(0, 3, 2, 1).reshape(n_pts * n_views, 2 * n + 1, 3)

        # pointflow on hypothesis pt
        offset_prob = self.decoder(x_sparse, pts_hyp, pts_feat, pts_batch_idx)
        offset_vals = torch.linspace(-n * offset, n * offset, 2 * n + 1) \
            .type_as(offset_prob).unsqueeze(0) \
            .expand(n_pts * n_views, 2 * n + 1)

        depth_offset = torch.sum(offset_vals * offset_prob, dim=1).view(n_views, *depthmap_current_scale.shape[1:])
        return depth_offset

    def forward(self, features_2d,
                inputs, outputs, dbatch=None,
                depth_pred_dict=None, depth_gt_dict=None,
                depth_batch_idx=None, offsets=[0.05, 0.05, 0.025]):
        """ Batch size set as 1 for one gpu card. Meaning only 1 fragment data is processed at a time.
        @var scale: denotes the voxel relative size, [2,1,0] as for [c,m,f]
        @return outputs: dict: {
            'coords: (Tensor), coordinates of voxels,
                     (number of voxels, 4) (4 : batch ind, x, y, z)
            'tsdf: (Tensor), TSDF of voxels,
                   (number of voxels, 1)
        }
        @return loss_dict: dict: {
            'tsdf_occ_loss_X: (Tensor), multi level loss
        }
        @return depth_offset_list: list with offset depth prediction in three scales
        """
        bs = 1
        pre_feat = None
        pre_coords = None
        loss_dict = {}
        loss_semseg_dict = {}
        loss_depth_dict = {}

        # ----coarse to fine----
        for i in range(self.cfg.N_STAGE):
            scale = self.cfg.N_STAGE - 1 - i
            interval = 2 ** scale

            if self.cfg.CDR.SEMSEG_REFMNT:
                links_current_stage = inputs['links_list'][scale].squeeze(0).clone()
                sparse_gt_coords = inputs['pth_voxelize_coords_list'][scale].squeeze(0)
                sparse_gt_dummy_feats = torch.zeros_like(sparse_gt_coords[:, 0]).unsqueeze(-1).type_as(sparse_gt_coords)
                sparse_3d_gt_feat = ME.SparseTensor(sparse_gt_dummy_feats, sparse_gt_coords.int())

            if self.cfg.DEPTH_PREDICTION:
                feats = features_2d[scale]  # (n_views, 1, backbone_2d_tensor[scale]==(c, h, w))
            else:
                feats = torch.stack([feat[scale] for feat in features_2d])

            if i == 0:
                # ----generate new coords----
                init_coords = generate_grid(self.cfg.N_VOX, interval)[0]  # [1, 3, -1]
                up_coords = []
                for b in range(bs):
                    # make the up_coords[b][0,:] with the batch idx
                    up_coords.append(
                        torch.cat((torch.ones(1, init_coords.shape[-1]).to(init_coords.device) * b, init_coords),
                                  dim=0))

                # if multiple batch, cast up_coords list into tensor from [4, n] to [n, 4]
                up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()
            else:
                # ----upsample coords for m/f stage----
                up_feat, up_coords, depth_pred_dict = self.upsample(pre_feat, pre_coords, interval,
                                                                    depth_pred_dict, i, self.cfg.STAGE_LIST, feats,
                                                                    depth_enabled=self.cfg.DEPTH_PREDICTION)

            # ----back project current stage feat of all frames in the fragment/batch----
            KRcam = inputs['proj_matrices'][:, :, scale].permute(1, 0, 2, 3).contiguous()
            volume, count = back_project(up_coords, inputs['vol_origin_partial'],
                                         self.cfg.VOXEL_SIZE, feats, KRcam)
            grid_mask = count > 1

            # ----concat feature from prev stage----
            if i != 0:
                feat = torch.cat([volume, up_feat], dim=1)
            else:
                feat = volume

            if not self.cfg.FUSION.FUSION_ON:
                tsdf_target, occ_target = self.get_target(up_coords, inputs, scale)

            # ----convert to aligned camera coordinate----
            r_coords = up_coords.detach().clone().float()
            for b in range(bs):
                batch_ind = torch.nonzero(up_coords[:, 0] == b).squeeze(1)
                coords_batch = up_coords[batch_ind][:, 1:].float()
                coords_batch = coords_batch * self.cfg.VOXEL_SIZE + inputs['vol_origin_partial'][b].float()
                coords_batch = torch.cat((coords_batch, torch.ones_like(coords_batch[:, :1])), dim=1)
                coords_batch = coords_batch @ inputs['world_to_aligned_camera'][b, :3, :].permute(1, 0).contiguous()
                r_coords[batch_ind, 1:] = coords_batch

            # batch index is in the last position
            r_coords = r_coords[:, [1, 2, 3, 0]]

            # ----sparse conv 3d backbone----
            point_feat = PointTensor(feat, r_coords)
            feat = self.sp_convs[i](point_feat)

            # ----gru fusion----
            if self.cfg.FUSION.FUSION_ON:
                up_coords, feat, tsdf_target, occ_target = \
                    self.gru_fusion(up_coords, feat, inputs, scale)
                grid_mask = torch.ones_like(feat[:, 0]).bool()

                # ----anchor implicit feature refinement----
                if self.cfg.CDR.FEAT_REFMNT and self.training:
                    feat = \
                        self.anchor_feature_decoder[scale](
                            depth_pred_dict[f'depth_{self.cfg.STAGE_LIST[i]}'],
                            dbatch.images.shape[-2:],
                            dbatch.rotmats,
                            dbatch.tvecs, dbatch.K,
                            scale,
                            feat=feat,
                            up_coords=up_coords,
                            interval=interval,
                            # tsdf_target=tsdf_target,
                            origin=inputs['vol_origin_partial'][0],
                            visualize_depth_backproj=self.cfg.VIS_DEBUG_REFMNT
                        )
            semseg_target = self.get_sparse_semseg_targets(up_coords, inputs, scale)

            # ----tsdf/occ/semseg preds----
            tsdf = self.tsdf_preds[i](feat)
            occ = self.occ_preds[i](feat)
            semseg_sparse = self.semseg_pred(feat, stage_idx=i)

            # ----point flow depth offset to refine coarse depth----
            if self.cfg.DEPTH_PREDICTION:
                loss = 0  # necessary to have a new assignment to overlap the previous loss val
                if i == 0:
                    # create input sparse dict for ptflow depth & feat offset, using pre_feat
                    sparse_feat_dict = self.process_sparse_feat_to_dict(sparse_geo_feats=feat,
                                                                        sparse_coords=up_coords,
                                                                        interval=interval)
                    for offset in offsets:
                        depth_offset_pred = self.depth_offset_and_feat_refmnt_wth_pointflow(
                            sparse_feat_dict,
                            depth_pred_dict[f'depth_{self.cfg.STAGE_LIST[i]}'],
                            depth_batch_idx,
                            feats.squeeze(1),
                            # (n_view, c, h, w) of the current scale feat
                            scale,
                            dbatch.rotmats, dbatch.tvecs, dbatch.K,
                            dbatch.ref_src_edges, dbatch.images.shape[-2:],
                            offset, 3)
                        depth_pred_dict[
                            f'depth_{self.cfg.STAGE_LIST[i]}'] += depth_offset_pred  # if you want inplace +=, need no grad for leaf node

                # visualizing depth prediction
                if self.cfg.DEPTH_PREDICTION and self.cfg.VIS_DEPTH and i == 2:
                    depth_fine_before_propagate = depth_pred_dict[f'depth_fine'].clone()

                depth_pred_dict[f'depth_{self.cfg.STAGE_LIST[i]}'] = \
                    self.depth_propagator[i](feats[self.ref_idx].squeeze(1),
                                             depth_pred_dict[f'depth_{self.cfg.STAGE_LIST[i]}'].unsqueeze(1)).squeeze(1)
                if depth_gt_dict is not None:
                    loss += self.compute_depth_loss(depth_pred_dict[f'depth_{self.cfg.STAGE_LIST[i]}'],
                                                    depth_gt_dict[f'depth_{self.cfg.STAGE_LIST[i]}'],
                                                    self.cfg.DEPTH_MVS.DEPTH_INTERVAL)

                loss_depth_dict[f'loss_depth_offset_{self.cfg.STAGE_LIST[i]}'] = loss

            # -------compute losses-------
            if tsdf_target is not None:
                loss = self.compute_loss(tsdf, occ, tsdf_target, occ_target,
                                         mask=grid_mask,
                                         pos_weight=self.cfg.POS_WEIGHT)
            else:
                loss = torch.Tensor(np.array([0]))[0]
            loss_dict.update({f'tsdf_occ_loss_{i}': loss})

            # ------define the sparsity for the next stage-----
            occupancy = occ.squeeze(1) > self.cfg.THRESHOLDS[i]
            occupancy[grid_mask == False] = False

            # visualizing depth prediction
            if self.cfg.DEPTH_PREDICTION and self.cfg.VIS_DEPTH and scale == 0:
                vis_2D_depth_prediction_comparison(
                    depth_fine_before_propagate,
                    depth_pred_dict[f'depth_{self.cfg.STAGE_LIST[i]}'].clone(),
                    dbatch.depth_images,
                    inputs['depth_from_mesh']
                )
                print('vis depth')

            num = int(occupancy.sum().data.cpu())

            # ------avoid out of memory: sample points if num of points is too large-----
            if self.training and num > self.cfg.TRAIN_NUM_SAMPLE[i] * bs:
                choice = np.random.choice(num, num - self.cfg.TRAIN_NUM_SAMPLE[i] * bs,
                                          replace=False)
                ind = torch.nonzero(occupancy)
                occupancy[ind[choice]] = False

            # sparsify
            pre_coords = up_coords[occupancy]
            pre_feat = feat[occupancy]
            pre_tsdf = tsdf[occupancy]
            pre_occ = occ[occupancy]
            pre_semseg = semseg_sparse[occupancy]

            if self.cfg.CDR.SEMSEG_REFMNT and self.cfg.DEPTH_PREDICTION:
                # ----Linking c/m/f stages----
                n_view, b, c, h, w = feats.shape

                # scaling resolution from img res to current feat res
                links_current_stage[:, 1:3, :] = \
                    (h / self.img_size_for_linking[1] * links_current_stage[:, 1:3, :].float()).int()

                sparse_feat_coords = torch.cat((torch.zeros(up_coords.shape[0], 1, dtype=torch.int).cuda(),
                                                up_coords[:, 1:].int()), dim=1)
                sparse_feat = ME.SparseTensor(feat, sparse_feat_coords,
                                              tensor_stride=2 ** scale,
                                              coordinate_manager=sparse_3d_gt_feat.coordinate_manager)

                # linked feat get 2d feats fused with the current sparse 3d feat out of sparse conv 3d backbone
                linked_feat = self.linkers[i](feat_2d_all=feats,
                                              # feat_3d=sparse_pre_feat,  # need to make this me.sparsetensor
                                              sparse_feat_3d=sparse_feat,
                                              links=links_current_stage,
                                              init_3d=sparse_3d_gt_feat,
                                              scale=scale,
                                              iter=iter)
                semseg_feat_3d = self.easy_mink_decoder[i](linked_feat).F
                # semseg_sparse = self.semseg_pred(semseg_feat_3d, i)  # semseg prediction

            # -------compute semseg losses, w/t or w/o semseg refinement-------
            if semseg_target is not None:
                loss_semseg = self.compute_semseg_occupied_loss(pre_semseg, semseg_target[occupancy])
            else:
                loss_semseg = torch.Tensor(np.array([0]))[0]
            loss_semseg_dict.update({f'semseg_occ_loss_{i}': loss_semseg})

            # examine occupancy
            if num == 0:
                logger.warning('no valid points: stage {}, occupancy is 0'.format(i))
                if self.cfg.DEPTH_PREDICTION:
                    return outputs, loss_dict, loss_semseg_dict, loss_depth_dict, depth_pred_dict
                return outputs, loss_dict, loss_semseg_dict
            for b in range(bs):
                batch_ind = torch.nonzero(pre_coords[:, 0] == b).squeeze(1)
                if len(batch_ind) == 0:  # pre_coords is None with occupancy
                    logger.warning('no valid points with occupancy: stage {}, batch {}'.format(i, b))
                    return outputs, loss_dict, loss_semseg_dict

            pre_feat = torch.cat([pre_feat, pre_tsdf, pre_occ], dim=1)

            if scale == 0:
                outputs['coords'] = pre_coords
                outputs['tsdf'] = pre_tsdf
                outputs['semseg'] = pre_semseg

        if self.cfg.DEPTH_PREDICTION:
            return outputs, loss_dict, loss_semseg_dict, loss_depth_dict, depth_pred_dict
        else:
            return outputs, loss_dict, loss_semseg_dict

    @staticmethod
    def compute_loss(tsdf, occ, tsdf_target, occ_target, loss_weight=(1, 1),
                     mask=None, pos_weight=1.0):
        """
        @param tsdf: (Tensor), predicted tsdf, (N, 1)
        @param occ: (Tensor), predicted occupancy, (N, 1)
        @param tsdf_target: (Tensor),ground truth tsdf, (N, 1)
        @param occ_target: (Tensor), ground truth occupancy, (N, 1)
        @param loss_weight: (Tuple)
        @param mask: (Tensor), mask voxels which cannot be seen by all views
        @param pos_weight: (float)
        @return: loss: (Tensor)
        """
        # compute occupancy/tsdf loss
        tsdf = tsdf.view(-1)
        occ = occ.view(-1)
        tsdf_target = tsdf_target.view(-1)
        occ_target = occ_target.view(-1)
        if mask is not None:
            mask = mask.view(-1)
            tsdf = tsdf[mask]
            occ = occ[mask]
            tsdf_target = tsdf_target[mask]
            occ_target = occ_target[mask]

        n_all = occ_target.shape[0]
        n_p = occ_target.sum()
        if n_p == 0:
            logger.warning('target: no valid voxel when computing loss')
            return torch.Tensor([0.0]).cuda()[0] * tsdf.sum()
        w_for_1 = (n_all - n_p).float() / n_p
        w_for_1 *= pos_weight

        # compute occ bce loss
        occ_loss = F.binary_cross_entropy_with_logits(occ, occ_target.float(), pos_weight=w_for_1)

        # compute tsdf l1 loss
        tsdf = apply_log_transform(tsdf[occ_target])
        tsdf_target = apply_log_transform(tsdf_target[occ_target])
        tsdf_loss = torch.mean(torch.abs(tsdf - tsdf_target))

        # compute final loss
        loss = loss_weight[0] * occ_loss + loss_weight[1] * tsdf_loss
        return loss

    @staticmethod
    def compute_semseg_occupied_loss(semseg_sparse_occupied, semseg_trgt_occupied):
        semseg_trgt = semseg_trgt_occupied.view(-1)
        semsemg_sparse = semseg_sparse_occupied.permute(1, 0).unsqueeze(0)
        semseg_trgt = semseg_trgt.unsqueeze(0).to(torch.int64)
        semseg_loss = F.cross_entropy(semsemg_sparse, semseg_trgt, reduction='none', ignore_index=-1).mean()

        return semseg_loss

    @staticmethod
    def compute_depth_loss(depth_pred, depth_gt, depth_interval):
        """compute mean abs error as depth loss"""
        if depth_gt.shape != depth_pred.shape:
            depth_gt = F.interpolate(depth_gt.unsqueeze(1), depth_pred.shape[-2:],
                                     mode='nearest').squeeze(1)
        mask_valid = (~torch.eq(depth_gt, 0.0)).type(torch.float)
        denom = torch.sum(mask_valid, dim=(1, 2)) + 1e-7
        masked_abs_error = mask_valid * torch.abs(depth_pred - depth_gt)
        masked_mae = torch.sum(masked_abs_error, dim=(1, 2))
        masked_mae = torch.mean((masked_mae / depth_interval) / denom)
        return masked_mae
