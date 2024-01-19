import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsparse.tensor import PointTensor
from utils import sparse_to_dense_channel, sparse_to_dense_torch
from .modules import ConvGRU


class GRUFusion(nn.Module):
    """
    Two functionalities of this class:
    1. GRU Fusion module as in the paper. Update hidden state features with ConvGRU.
    2. Input geometric and semantic feature in a ping pong manner.
    2. Substitute TSDF in the global volume when direct_substitute = True for output.
    """

    def __init__(self, cfg, ch_in=None, direct_substitute=False):
        super(GRUFusion, self).__init__()
        self.cfg = cfg
        self.direct_substitute = direct_substitute

        if direct_substitute:
            # tsdf
            self.ch_in = [1, 1, 1]
            self.feat_init = 1
            self.ch_in_semseg = cfg.CDR.SEMSEG_CLASS_3D  # for each stage, #class is configured
        else:
            # features
            self.ch_in = ch_in
            self.feat_init = 0

        self.n_scales = len(cfg.THRESHOLDS) - 1
        self.scene_name = [None, None, None]
        self.global_origin = [None, None, None]
        self.global_volume = [None, None, None]
        self.target_tsdf_volume = [None, None, None]  # global target tsdf vol
        self.global_volume_semseg = [None, None, None]

        if direct_substitute:
            self.fusion_nets = None
        else:
            self.fusion_nets = nn.ModuleList()
            for i, ch in enumerate(ch_in):
                self.fusion_nets.append(ConvGRU(hidden_dim=ch,
                                                input_dim=ch,
                                                pres=1,
                                                vres=self.cfg.VOXEL_SIZE * 2 ** (self.n_scales - i)))

    def reset(self, i):
        self.global_volume[i] = PointTensor(torch.Tensor([]), torch.Tensor([]).view(0, 3).long()).cuda()
        self.target_tsdf_volume[i] = PointTensor(torch.Tensor([]), torch.Tensor([]).view(0, 3).long()).cuda()
        self.global_volume_semseg[i] = torch.Tensor([]).cuda()

    def convert2dense(self, current_coords, current_values, coords_target_global, tsdf_target, relative_origin,
                      scale):
        """Update tsdf/occ target.

        1. convert sparse feature to dense feature;
        2. combine current feature coordinates and previous coordinates within FBV from global hidden state to get
        new feature coordinates (updated_coords);
        3. fuse ground truth tsdf.

        @param current_coords: (Tensor), current coordinates, (N, 3)
        @param current_values: (Tensor), current features/tsdf, (N, C), when fuse_to_global is tsdf
        @param coords_target_global: (Tensor), ground truth coordinates, (N', 3)
        @param tsdf_target: (Tensor), tsdf ground truth, (N',)
        @param relative_origin: (Tensor), origin in global volume, (3,)
        @param scale:
        @return: updated_coords: (Tensor), coordinates after combination, (N', 3)
        @return: current_volume: (Tensor), current dense feature/tsdf volume, (DIM_X, DIM_Y, DIM_Z, C)
        @return: global_volume: (Tensor), global dense feature/tsdf volume, (DIM_X, DIM_Y, DIM_Z, C)
        @return: target_volume: (Tensor), dense target tsdf volume, (DIM_X, DIM_Y, DIM_Z, 1)
        @return: valid: mask: 1 represent in current FBV (N,)
        @return: valid_target: gt mask: 1 represent in current FBV (N,)
        """
        # previous frame
        global_coords = self.global_volume[scale].C
        global_value = self.global_volume[scale].F
        global_tsdf_target = self.target_tsdf_volume[scale].F
        global_coords_target = self.target_tsdf_volume[scale].C

        dim = (torch.div(torch.Tensor(self.cfg.N_VOX).cuda(), 2 ** scale)).int()
        dim_list = dim.data.cpu().numpy().tolist()

        # mask voxels that are out of the FBV
        global_coords = global_coords - relative_origin
        valid = ((global_coords < dim) & (global_coords >= 0)).all(dim=-1)

        # sparse to dense
        global_volume = sparse_to_dense_channel(global_coords[valid], global_value[valid], dim_list,
                                                self.ch_in[2 - scale],
                                                self.feat_init, global_value.device)
        current_volume = sparse_to_dense_channel(current_coords, current_values, dim_list, self.ch_in[2 - scale],
                                                 self.feat_init, global_value.device)

        # change the structure of sparsity, combine current coordinates and previous coordinates from global volume
        if self.direct_substitute:
            updated_coords = torch.nonzero((global_volume.abs() < 1).any(-1) | (current_volume.abs() < 1).any(-1))
        else:
            updated_coords = torch.nonzero((global_volume != 0).any(-1) | (current_volume != 0).any(-1))

        # fuse ground truth
        if tsdf_target is not None:
            # mask voxels that are out of the FBV
            global_coords_target = global_coords_target - relative_origin
            valid_target = ((global_coords_target < dim) & (global_coords_target >= 0)).all(dim=-1)
            # combine current tsdf and global tsdf
            coords_target = torch.cat([global_coords_target[valid_target], coords_target_global])[:, :3]
            tsdf_target = torch.cat([global_tsdf_target[valid_target], tsdf_target.unsqueeze(-1)])
            target_volume = sparse_to_dense_channel(coords_target, tsdf_target, dim_list, 1, 1,
                                                    tsdf_target.device)
        else:
            target_volume = valid_target = None

        return updated_coords, current_volume, global_volume, target_volume, valid, valid_target

    def convert2dense_fuse2global(self, current_coords, current_values, coords_target_global,
                                  tsdf_target, relative_origin, scale, current_values_semseg):
        """ Get semseg volume for output.
        1. convert sparse feature to dense feature;
        2. combine current feature coordinates and previous coordinates within FBV from global hidden state to get
        new feature coordinates (updated_coords);
        3. fuse ground truth tsdf.

        @param current_coords: (Tensor), current coordinates, (N, 3)
        @param current_values: (Tensor), current features/tsdf, (N, C), when fuse_to_global is tsdf
        @param current_values_semseg: Tensor, currect semseg vol, for fuse_to_global
        @param coords_target_global: (Tensor), ground truth coordinates, (N', 3)
        @param tsdf_target: (Tensor), tsdf ground truth, (N',)
        @param relative_origin: (Tensor), origin in global volume, (3,)
        @param scale: resolution scale.
        @return: updated_coords: (Tensor), coordinates after combination, (N', 3)
        @return: current_volume: (Tensor), current dense feature/tsdf volume, (DIM_X, DIM_Y, DIM_Z, C)
        @return: global_volume: (Tensor), global dense feature/tsdf volume, (DIM_X, DIM_Y, DIM_Z, C)
        @return: target_volume: (Tensor), dense target tsdf volume, (DIM_X, DIM_Y, DIM_Z, 1)
        @return: valid: mask: 1 represent in current FBV (N,)
        @return: valid_target: gt mask: 1 represent in current FBV (N,)
        """
        # previous frame
        global_coords = self.global_volume[scale].C
        global_value = self.global_volume[scale].F
        global_tsdf_target = self.target_tsdf_volume[scale].F
        global_coords_target = self.target_tsdf_volume[scale].C
        global_value_semseg = self.global_volume_semseg[scale]

        dim = (torch.div(torch.Tensor(self.cfg.N_VOX).cuda(), 2 ** scale, rounding_mode='trunc')).int()
        dim_list = dim.data.cpu().numpy().tolist()

        # mask voxels that are out of the FBV
        global_coords = global_coords - relative_origin
        valid = ((global_coords < dim) & (global_coords >= 0)).all(dim=-1)

        # sparse to dense
        global_volume = sparse_to_dense_channel(global_coords[valid], global_value[valid], dim_list,
                                                self.ch_in[2 - scale],
                                                self.feat_init, global_value.device)
        current_volume = sparse_to_dense_channel(current_coords, current_values, dim_list, self.ch_in[2 - scale],
                                                 self.feat_init, global_value.device)
        global_volume_semseg = sparse_to_dense_channel(global_coords[valid], global_value_semseg[valid], dim_list,
                                                       self.ch_in_semseg,
                                                       self.feat_init, global_value_semseg.device)
        current_volume_semseg = sparse_to_dense_channel(current_coords, current_values_semseg, dim_list,
                                                        self.ch_in_semseg,
                                                        self.feat_init, global_value_semseg.device)

        if self.direct_substitute:
            updated_coords = torch.nonzero((global_volume.abs() < 1).any(-1) | (current_volume.abs() < 1).any(-1))
        else:
            updated_coords = torch.nonzero((global_volume != 0).any(-1) | (current_volume != 0).any(-1))

        # fuse ground truth
        if tsdf_target is not None:
            # mask voxels that are out of the FBV
            global_coords_target = global_coords_target - relative_origin
            valid_target = ((global_coords_target < dim) & (global_coords_target >= 0)).all(dim=-1)
            # combine current gt tsdf and global gt tsdf
            coords_target = torch.cat([global_coords_target[valid_target], coords_target_global])[:, :3]
            tsdf_target = torch.cat([global_tsdf_target[valid_target], tsdf_target.unsqueeze(-1)])
            # sparse to dense
            target_volume = sparse_to_dense_channel(coords_target, tsdf_target, dim_list, 1, 1,
                                                    tsdf_target.device)
        else:
            target_volume = valid_target = None

        return updated_coords, current_volume, global_volume, target_volume, valid, valid_target, current_volume_semseg, global_volume_semseg

    def update_map(self, value, coords, target_volume, valid, valid_target,
                   relative_origin, scale, semseg_value=None):
        """
        Replace Hidden state/tsdf in global Hidden state/tsdf volume by direct substitute corresponding voxels
        @param value: (Tensor) fused feature (N, C)
        @param coords: (Tensor) updated coords (N, 3)
        @param target_volume: (Tensor) tsdf volume (DIM_X, DIM_Y, DIM_Z, 1)
        @param valid: (Tensor) mask: 1 represent in current FBV (N,)
        @param valid_target: (Tensor) gt mask: 1 represent in current FBV (N,)
        @param relative_origin: (Tensor), origin in global volume, (3,)
        @param scale:
        @param semseg_value: Tensor, fused semseg feat (N, C). Not in gru_fusion, but in fuse_to_global() after
                semseg prediction.
        @return:
        """
        # pred
        self.global_volume[scale].F = torch.cat(
            [self.global_volume[scale].F[valid == False], value])
        coords = coords + relative_origin
        self.global_volume[scale].C = torch.cat([self.global_volume[scale].C[valid == False], coords])
        if semseg_value is not None:
            self.global_volume_semseg[scale] = torch.cat(
                [self.global_volume_semseg[scale][valid == False], semseg_value])

        # target
        if target_volume is not None:
            target_volume = target_volume.squeeze()
            self.target_tsdf_volume[scale].F = torch.cat(
                [self.target_tsdf_volume[scale].F[valid_target == False],
                 target_volume[target_volume.abs() < 1].unsqueeze(-1)])
            target_coords = torch.nonzero(target_volume.abs() < 1) + relative_origin

            self.target_tsdf_volume[scale].C = torch.cat(
                [self.target_tsdf_volume[scale].C[valid_target == False], target_coords])

    def save_mesh(self, scale, outputs, scene):
        if outputs is None:
            outputs = dict()
        if "scene_name" not in outputs:
            outputs['origin'] = []
            outputs['scene_tsdf'] = []
            outputs['scene_name'] = []

            outputs['scene_semseg'] = []
            outputs['semseg_attribute'] = []

        # only keep the latest scene result
        if scene in outputs['scene_name']:
            # delete old
            idx = outputs['scene_name'].index(scene)
            del outputs['origin'][idx]
            del outputs['scene_tsdf'][idx]
            del outputs['scene_name'][idx]

            del outputs['scene_semseg'][idx]
            del outputs['semseg_attribute'][idx]

        # scene name
        outputs['scene_name'].append(scene)

        fuse_coords = self.global_volume[scale].C
        tsdf = self.global_volume[scale].F.squeeze(-1)
        max_c = torch.max(fuse_coords, dim=0)[0][:3]
        min_c = torch.min(fuse_coords, dim=0)[0][:3]
        outputs['origin'].append(min_c * self.cfg.VOXEL_SIZE * (2 ** scale))

        ind_coords = fuse_coords - min_c
        dim_list = (max_c - min_c + 1).int().data.cpu().numpy().tolist()
        tsdf_volume = sparse_to_dense_torch(ind_coords, tsdf, dim_list, 1, tsdf.device)
        outputs['scene_tsdf'].append(tsdf_volume)

        # all the semseg mod in neucon does not has trained param, it just make fuse_to_global() can output semseg mesh
        semseg = self.global_volume_semseg[scale].squeeze(-1)
        semseg_volume = sparse_to_dense_channel(ind_coords, semseg, dim_list, 41, 1, tsdf.device)
        outputs['scene_semseg'].append(semseg_volume)
        outputs['semseg_attribute'].append(semseg)

        return outputs

    def forward(self, coords, values_in, inputs, scale=0,  # fuse_to_global takes scale=0
                outputs=None, save_mesh=False, semseg_values_in=None):
        """
        @param coords: (Tensor), coordinates of voxels, (N, 4) (4 : Batch ind, x, y, z)
        @param values_in: (Tensor), features/tsdf, (N, C) (feat for gru_fusion, tsdf for fuse_to_global)
        @param semseg_values_in: (Tensor), semseg volume (just as tsdf in fuse_to_global), (N, C).
                only being used in updated_map().

                delete for now. Seems like in gru fusion, semseg no need to be gru together.
                because semseg and tsdf share the same coordinate. gru should take care of one only.
                But a good semseg target should be generated here.

        @param inputs: dict: meta data from dataloader
        @param scale: from 2 to 0, indicates coarse to fine, matched with tsdf gt, 2 to 0
        @param outputs:
        @param save_mesh: a bool to indicate whether or not to save the reconstructed mesh of current sample
        if direct_substitute:
        @return outputs: dict: {
            'origin':                  (List), origin of the predicted partial volume,
                                    [3]
            'scene_tsdf':              (List), predicted tsdf volume,
                                    [(nx, ny, nz)]
            'target':                  (List), ground truth tsdf volume,
                                    [(nx', ny', nz')]
            'scene_name':                  (List), name of each scene in 'scene_tsdf',
                                    [string]
            'scene_semseg'          predicted semseg volume,
                                    [(nx, ny, nz)]
        }
        else:
        @return updated_coords_all: (Tensor), updated coordinates, (N', 4) (4 : Batch ind, x, y, z)
        @return values_all: (Tensor), features after gru fusion, (N', C)
        @return tsdf_target_all: (Tensor), tsdf ground truth, (N', 1)
        @return occ_target_all: (Tensor), occupancy ground truth, (N', 1)
        """
        if self.global_volume[scale] is not None:
            # delete computational graph to save memory
            self.global_volume[scale] = self.global_volume[scale].detach()
        if self.global_volume_semseg[scale] is not None:
            self.global_volume_semseg[scale] = self.global_volume_semseg[scale].detach()

        batch_size = len(inputs['fragment'])
        interval = 2 ** scale

        # global vars for three stages
        tsdf_target_all = None
        occ_target_all = None
        values_all = None
        updated_coords_all = None

        # ---incremental fusion----
        for i in range(batch_size):
            scene = inputs['scene'][i]  # scene name
            global_origin = inputs['vol_origin'][i]  # origin of global volume
            origin = inputs['vol_origin_partial'][i]  # origin of part volume

            # save every global hidden state when new scene comes in thus scene_name is different
            # for inter-fragments recurrent fusion by read/write outputs
            if scene != self.scene_name[scale] and self.scene_name[scale] is not None and self.direct_substitute:
                outputs = self.save_mesh(scale, outputs, self.scene_name[scale])

            # if self.scene is empty or this fragment is from new scene, we reinitialize backend map
            if self.scene_name[scale] is None or scene != self.scene_name[scale]:
                self.scene_name[scale] = scene
                self.reset(scale)
                self.global_origin[scale] = global_origin

            # each level has its corresponding voxel size
            voxel_size = self.cfg.VOXEL_SIZE * interval

            # relative origin in global volume
            relative_origin = (origin - self.global_origin[scale]) / voxel_size
            relative_origin = relative_origin.cuda().long()

            # if in current batch, the 3d pt index
            batch_ind = torch.nonzero(coords[:, 0] == i).squeeze(1)
            if len(batch_ind) == 0:
                # nothing to be fused if not in current batch
                continue
            coords_b = torch.div(coords[batch_ind, 1:].long(), interval, rounding_mode='trunc')
            values = values_in[batch_ind]  # access batch_ind, since only batch 0, so values=values_in

            if self.direct_substitute:
                semseg_values = semseg_values_in[batch_ind]

            if 'occ_list' in inputs.keys():
                # get partial gt
                occ_target = inputs['occ_list'][scale][i]
                # sparse tsdf target for the whole scene for current scale
                tsdf_target = inputs['tsdf_list'][scale][i][occ_target]
                coords_target = torch.nonzero(occ_target)
            else:
                coords_target = tsdf_target = None

            # convert to dense:
            # ctf updates global_target for tsdf/occ
            if not self.direct_substitute:
                updated_coords, current_volume, global_volume, target_volume, valid, valid_target = self.convert2dense(
                    coords_b,
                    values,
                    coords_target,
                    tsdf_target,
                    relative_origin,
                    scale)
            else:
                # fuse_to_global creates tsdf/semseg vol
                updated_coords, current_volume, global_volume, target_volume, valid, valid_target, \
                current_volume_semseg, global_volume_semseg = self.convert2dense_fuse2global(
                    coords_b,
                    values,
                    coords_target,
                    tsdf_target,
                    relative_origin,
                    scale,
                    semseg_values
                )

            # dense to sparse: get features using new feature coordinates (updated_coords) within current fragment
            # for gru fusion
            values = current_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]
            global_values = global_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]
            if self.direct_substitute:  # only in fuse_to_global, value is tsdf, and semseg_value is needed
                semseg_values = current_volume_semseg[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]
                global_volume_semseg = global_volume_semseg[
                    updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]

            # get fused gt according to the updated x,y,z coords
            if not self.direct_substitute:
                if target_volume is not None:
                    tsdf_target = target_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]
                    occ_target = tsdf_target.abs() < 1
            else:
                tsdf_target = occ_target = None

            if not self.direct_substitute:
                # convert to aligned camera coordinate
                r_coords = updated_coords.detach().clone().float()
                r_coords = r_coords.permute(1, 0).contiguous().float() * voxel_size + origin.unsqueeze(-1).float()
                r_coords = torch.cat((r_coords, torch.ones_like(r_coords[:1])), dim=0)
                r_coords = inputs['world_to_aligned_camera'][i, :3, :] @ r_coords
                r_coords = torch.cat([r_coords, torch.zeros(1, r_coords.shape[-1]).to(r_coords.device)])
                r_coords = r_coords.permute(1, 0).contiguous()

                h = PointTensor(global_values, r_coords)  # hidden state
                x = PointTensor(values, r_coords)  # 3D geometric features

                values = self.fusion_nets[2 - scale](h, x)  # values is feat
                # only update 3d feat map output from GRU
                self.update_map(values, updated_coords, target_volume, valid, valid_target, relative_origin, scale)
            else:
                # fuse to global
                self.update_map(values, updated_coords, target_volume, valid, valid_target, relative_origin, scale,
                                semseg_value=semseg_values)

            if updated_coords_all is None:  # namely the coarse stage
                updated_coords_all = torch.cat([torch.ones_like(updated_coords[:, :1]) * i, updated_coords * interval],
                                               dim=1)
                values_all = values
                tsdf_target_all = tsdf_target
                occ_target_all = occ_target
            else:
                updated_coords = torch.cat([torch.ones_like(updated_coords[:, :1]) * i, updated_coords * interval],
                                           dim=1)
                updated_coords_all = torch.cat([updated_coords_all, updated_coords])
                values_all = torch.cat([values_all, values])
                if tsdf_target_all is not None:
                    tsdf_target_all = torch.cat([tsdf_target_all, tsdf_target])
                    occ_target_all = torch.cat([occ_target_all, occ_target])

            # only true with fuse_to_global()
            if self.direct_substitute and save_mesh:
                outputs = self.save_mesh(scale, outputs, self.scene_name[scale])

        if self.direct_substitute:
            return outputs
        else:
            # output 3D feature
            return updated_coords_all, values_all, tsdf_target_all, occ_target_all
