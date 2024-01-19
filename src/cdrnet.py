import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ctf_fragnet import CoarseToFineFragNet
from src.gru_fusion import GRUFusion
from src.backbone import MnasMulti
from src.semseg_heads import SemSegHead2D
from src.cost_reg import CostRegNet
from ops.geometry import batched_build_plane_sweep_volume_tensor
from utils import tocuda
from torch_scatter import scatter


class CrossDimensionalRefmntNet(nn.Module):
    """
    CDRNet main class.
    """

    def __init__(self, cfg):
        super(CrossDimensionalRefmntNet, self).__init__()
        self.cfg = cfg.MODEL
        self.logdir = cfg.LOGDIR
        alpha = float(self.cfg.BACKBONE2D.ARC.split('-')[-1])

        # other hparams
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        # mvs configs
        self.feat_dim = cfg.MODEL.CDR.CHANNEL_FEAT_DIM
        self.img_size = cfg.MODEL.CDR.IMG_SIZE

        # networks enabled by configs
        self.backbone2d = MnasMulti(alpha=alpha)
        if self.cfg.DEPTH_PREDICTION:
            if self.cfg.CDR.SEMSEG_2D:
                self.semseg_pred_2d = SemSegHead2D(cfg.MODEL.BACKBONE2D.CHANNELS[-1],
                                                   cfg.MODEL.CDR.SEMSEG_CLASS_2D)
            if self.cfg.CDR.DEPTH_PRED:
                self.cost_reg_cnn = CostRegNet(24, 8)

        self.cdrnet = CoarseToFineFragNet(cfg.MODEL)
        self.fuse_to_global = GRUFusion(cfg.MODEL, direct_substitute=True)

    def normalizer(self, x):
        """ Normalizes the RGB imported to the input range"""
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)

    def depth_pred_mvs_on_backbone2d(self, input_imgs, dbatch=None):
        """
        Initial depth prediction coarse from mvs is 56x56.

        Cfg True then do mvs, otherwise return feat
        """
        feats = [self.backbone2d(self.normalizer(img)) for img in input_imgs]
        depth_pred = None
        depth_start = self.cfg.DEPTH_MVS.DEPTH_START
        depth_interval = self.cfg.DEPTH_MVS.DEPTH_INTERVAL
        n_planes = self.cfg.DEPTH_MVS.N_INTERVALS
        depth_pred_size = self.cfg.DEPTH_MVS.DEPTH_PRED_SIZE

        if self.cfg.CDR.DEPTH_PRED and self.cfg.DEPTH_PREDICTION:
            quarter_list = [feat[0] for feat in feats]
            feats_quarter = torch.stack(quarter_list, dim=0).squeeze(1)

            # depth hypothesis
            depth_end = depth_start + depth_interval * (n_planes - 1)
            n_imgs, n_channels, img_height, img_width = dbatch.images.shape
            ref_idx, gather_idx = torch.unique(dbatch.ref_src_edges[0], return_inverse=True)
            n_ref_imgs = len(ref_idx)

            # warping the 2D feat with fp plane using all ref src pairs in dbatch
            with torch.no_grad():
                # fronto parallel volume for each frame with their pose
                pts_fp_vol = batched_build_plane_sweep_volume_tensor(dbatch.rotmats, dbatch.tvecs, dbatch.K, depth_start, depth_interval, n_planes, self.cfg.CDR.IMG_SIZE, depth_pred_size)
                pts_fp_vol = pts_fp_vol.type_as(dbatch.images)
                n_pts = pts_fp_vol.shape[2]
                w = torch.ones((n_imgs, 1, n_pts), dtype=torch.float32).type_as(dbatch.images)
                # homogenous coordinates
                pts_fp_homo = torch.cat((pts_fp_vol, w), dim=1)
                # prj mat from dbatch cam coord to img plane, unsqueeze it is fatser than tensor(None)
                P = torch.cat((dbatch.rotmats, dbatch.tvecs.unsqueeze(-1)), dim=2)
                P = torch.bmm(dbatch.K, P)
                # src backbone2d feat warping into ref fp vol, with src prj mat
                pts_src_to_ref_homo = torch.bmm(P[dbatch.ref_src_edges[1]], pts_fp_homo[dbatch.ref_src_edges[0]])
                z_buffer = pts_src_to_ref_homo[:, 2]
                z_buffer = torch.abs(z_buffer) + 1e-8  # to prevent div/0
                # cart coords
                pts_src_to_ref = pts_src_to_ref_homo[:, :2] / z_buffer.unsqueeze(1)
                # with ref_src_pairs grids, each grid is a fp plane
                grid = pts_src_to_ref.transpose(2, 1).view(pts_src_to_ref.shape[0], n_pts, 1, 2)
                # norm to [-1, 1] according to scale
                grid[..., 0] = (grid[..., 0] / float(self.cfg.CDR.IMG_SIZE[1] - 1)) * 2 - 1.0  # x
                grid[..., 1] = (grid[..., 1] / float(self.cfg.CDR.IMG_SIZE[0] - 1)) * 2 - 1.0  # y

            # feat fetching by differential homography
            # src backbone2d feat warping to ref fp vol, to create feat vol
            x_vox = F.grid_sample(feats_quarter[dbatch.ref_src_edges[1]], grid, mode='bilinear', align_corners=True)
            x_vox = x_vox.squeeze(3).view(-1, self.cfg.CDR.CHANNEL_FEAT_DIM, n_planes, *depth_pred_size)

            # var cost aggregation
            x_avg = scatter(x_vox, gather_idx, dim=0, reduce='mean', dim_size=n_ref_imgs)
            x_avg_sq = scatter(x_vox ** 2, gather_idx, dim=0, reduce='mean', dim_size=n_ref_imgs)
            x_var = x_avg_sq - x_avg ** 2

            # cost vol regularization
            x_reg = self.cost_reg_cnn(x_var).squeeze(1)
            x_prob = F.softmax(-x_reg, dim=1)

            # coarse depth pred
            depth_vals = torch.linspace(depth_start, depth_end, n_planes).type_as(dbatch.images)
            depth_volume = depth_vals.unsqueeze(0).repeat(n_ref_imgs, 1)
            depth_volume = depth_volume.view(n_ref_imgs, n_planes, 1, 1).expand(x_prob.shape)
            depth_pred = torch.sum(depth_volume * x_prob, dim=1)  # b x dw x dh

        return feats, depth_pred

    @staticmethod
    def depth_interpolation(stage_list, depth_pred_init, depth_gt, feats_2d):
        """This method performs both depth prediction and ground truth interpolation.
        feats_2d is from mvs on backbone 2d, expected as [feats_quarter, feats_eighth, feats_sixteenth]"""
        depth_dict = {}
        depth_gt_dict = {}
        depth_dict['depth_coarse'] = F.interpolate(depth_pred_init.unsqueeze(1), feats_2d[2].shape[-2:],
                                                   mode='nearest').squeeze(1)
        for i in range(len(stage_list)):
            depth_gt_dict[f'depth_{stage_list[i]}'] = F.interpolate(depth_gt.unsqueeze(1),
                                                                    feats_2d[2 - i].shape[-2:],
                                                                    mode='nearest').squeeze(1)

        return depth_dict, depth_gt_dict

    def forward(self, inputs, save_mesh=False):
        """

        :param inputs: dict: {
            'imgs':                    (Tensor), imported,
                                    (batch size, number of views, C, H, W)
            'vol_origin':              (Tensor), origin of the full voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'vol_origin_partial':      (Tensor), origin of the partial voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'world_to_aligned_camera': (Tensor), matrices: transform from world coords to aligned camera coords,
                                    (batch size, number of views, 4, 4)
            'proj_matrices':           (Tensor), projection matrix,
                                    (batch size, number of views, number of scales, 4, 4)
            when we have ground truth:
            'tsdf_list':               (List), tsdf ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            'occ_list':                (List), occupancy ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]

            fred appended this:
            'semseg_list':          (List), occupancy ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]


            others: unused in network
        }
        :param save_mesh: a bool to indicate whether or not to save the reconstructed mesh of current sample
        :return: outputs: dict: {
            'coords':                  (Tensor), coordinates of voxels,
                                    (number of voxels, 4) (4 : batch ind, x, y, z)
            'tsdf':                    (Tensor), TSDF of voxels,
                                    (number of voxels, 1)
            When it comes to save results:
            'origin':                  (List), origin of the predicted partial volume,
                                    [3]
            'scene_tsdf':              (List), predicted tsdf volume,
                                    [(nx, ny, nz)]
        }
                 loss_dict: dict: {
            'tsdf_occ_loss_X':         (Tensor), multi level loss
            'total_loss':              (Tensor), total loss
        }
        """
        inputs = tocuda(inputs)
        outputs = {}
        vis_metrics = {}
        # for depth
        depth_pred_dict = {}
        vis_metrics['initial_depth_metrics'] = {}
        vis_metrics['loss_depth_init'] = {}
        vis_metrics['loss_depth_offset'] = {}
        vis_metrics['subloss2d'] = {}
        imgs = torch.unbind(inputs['imgs'], 1)

        if self.cfg.DEPTH_PREDICTION:
            # image feature extraction & init depth prediction
            dbatch = inputs['dbatch']
            features, depth_pred_init_mvs = self.depth_pred_mvs_on_backbone2d(
                input_imgs=imgs,
                dbatch=dbatch
            )
            # list to tensor: from (n_view, 3) into (n_view * 3,) of tensor (b, c, h, w)
            features = [torch.stack([features[view][stage] for view in range(9)], dim=0) for stage in
                           range(3)]
        else:
            features = [self.backbone2d(self.normalizer(img)) for img in imgs]

        if self.cfg.DEPTH_PREDICTION:
            # loss that includes loss_depth_init and _lambda * loss_offset from pt flow
            loss_depth = torch.tensor(0., dtype=torch.float32, device=torch.device("cuda:0"),
                                      requires_grad=False)
            ref_idx = torch.unique(dbatch.ref_src_edges[0])
            depth_batch_idx = dbatch.images_batch_idx[ref_idx]  # current depth batch index

            # init depth supervision, gt size modified for metrics cal
            depth_gt_sm = F.interpolate(dbatch.depth_images.unsqueeze(1), depth_pred_init_mvs.shape[-2:],
                                        mode='nearest').squeeze(1)
            loss_depth_init = self.cdrnet.compute_depth_loss(depth_pred_init_mvs, depth_gt_sm, self.cfg.DEPTH_MVS.DEPTH_INTERVAL)
            metrics_2d_depth = compute_2d_depth_metrics(depth_pred_init_mvs, depth_gt_sm)
            vis_metrics['initial_depth_metrics'] = metrics_2d_depth
            vis_metrics['loss_depth_init'] = loss_depth_init
            loss_depth += loss_depth_init

            # list with three resolutions of depth gt (c/m/f) and depth_pred (from init to c) with interpolation
            depth_pred_dict, depth_gt_dict = self.depth_interpolation(self.cfg.STAGE_LIST, depth_pred_init_mvs,
                                                                      dbatch.depth_images, features)

            # no depth refinement, but semseg refinement
            outputs, loss_dict, loss_semseg_dict, loss_depth_dict, depth_pred_dict = \
                self.cdrnet(features_2d=features, inputs=inputs, outputs=outputs,
                            dbatch=dbatch,
                            depth_pred_dict=depth_pred_dict,
                            depth_gt_dict=depth_gt_dict,
                            depth_batch_idx=depth_batch_idx)
            if not self.training:
                depth_pred_dict['depth_init'] = depth_pred_init_mvs
        else:
            # coarse-to-fine decoder: SparseConv and GRU Fusion.
            # in: image feature; out: sparse coords and tsdf
            outputs, loss_dict, loss_semseg_dict = self.cdrnet(features, inputs, outputs)

        # fuse to global volume.
        if not self.training and 'coords' in outputs.keys():
            outputs = self.fuse_to_global(outputs['coords'], outputs['tsdf'], inputs,
                                          outputs=outputs, save_mesh=save_mesh,
                                          semseg_values_in=outputs['semseg'])

        weighted_loss = 0
        sub_tsdf_semseg_loss = {}
        total_tsdf_occ_loss = 0
        for i, (k, v) in enumerate(loss_dict.items()):
            weighted_loss += v * self.cfg.LW[i]
            total_tsdf_occ_loss += v * self.cfg.LW[i]
        vis_metrics['total_tsdf_occ_loss'] = total_tsdf_occ_loss
        sub_tsdf_semseg_loss.update(loss_dict)

        # for semseg ablation
        total_3d_semseg_loss = 0
        for i, (k, v) in enumerate(loss_semseg_dict.items()):
            weighted_loss += v * self.cfg.LW_SEMSEG[i]
            # * .1 coz tsdf loss is 10x larger than semseg loss
            total_3d_semseg_loss += v * self.cfg.LW_SEMSEG[i]
        vis_metrics['total_3d_semseg_loss'] = total_3d_semseg_loss
        sub_tsdf_semseg_loss.update(loss_semseg_dict)

        vis_metrics.update({'subloss3d': sub_tsdf_semseg_loss})

        total_loss_dict = {'total_loss': weighted_loss}
        vis_metrics.update(total_loss_dict)

        return outputs, vis_metrics


def compute_2d_depth_metrics(depth_pred, depth_gt, pred_valid=None, convert_to_cpu=False):
    out = {}
    with torch.no_grad():
        valid = (depth_gt >= .5) & (depth_gt < 65.)
        if pred_valid is not None:
            valid = valid & pred_valid
            v_prec = torch.mean(torch.sum(pred_valid, dim=(1, 2)) / (pred_valid.shape[1] * pred_valid.shape[2]))
            out['prec_valid'] = v_prec
        valid = valid.type(torch.float)
        denom = torch.sum(valid, dim=(1, 2)) + 1e-7
        abs_diff = torch.abs(depth_pred - depth_gt)
        abs_inv = torch.abs(1. / depth_pred - depth_gt)
        abs_inv[torch.isinf(abs_inv)] = 0.  # handle div /0
        abs_inv[torch.isnan(abs_inv)] = 0.

        abs_rel = torch.mean(torch.sum((abs_diff / (depth_gt + 1e-7)) * valid, dim=(1, 2)) / denom)
        sq_rel = torch.mean(torch.sum((abs_diff ** 2) / (depth_gt + 1e-7) * valid, dim=(1, 2)) / denom)
        rmse = torch.mean(torch.sqrt(torch.sum(abs_diff ** 2 * valid, dim=(1, 2)) / denom))
        abs_diff = torch.mean(torch.sum(abs_diff ** 2 * valid, dim=(1, 2)) / denom)
        abs_inv = torch.mean(torch.sum(abs_inv * valid, dim=(1, 2)) / denom)

        r1 = (depth_pred / depth_gt).unsqueeze(-1)
        r2 = (depth_gt / depth_pred).unsqueeze(-1)
        rel_max = torch.max(torch.cat((r1, r2), dim=-1), dim=-1)[0]

        # for sigma, max(depth/depth_gt), evaluation
        d_125 = torch.mean(torch.sum((rel_max < 1.25) * valid, dim=(1, 2)) / denom)
        d_125_2 = torch.mean(torch.sum((rel_max < 1.25 ** 2) * valid, dim=(1, 2)) / denom)
        d_125_3 = torch.mean(torch.sum((rel_max < 1.25 ** 3) * valid, dim=(1, 2)) / denom)

        out.update({
            'abs_rel': abs_rel,
            'abs_diff': abs_diff,
            'abs_inv': abs_inv,
            'sq_rel': sq_rel,
            'rmse': rmse,
            'd_125': d_125,
            'd_125_2': d_125_2,
            'd_125_3': d_125_3
        })
        if convert_to_cpu:
            out = {k: v.cpu().item() for k, v in out.items()}
        return out
