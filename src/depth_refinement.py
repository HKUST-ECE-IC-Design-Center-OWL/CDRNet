import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsparse.tensor as tst
import MinkowskiEngine as ME
import ops.geometry as geo
from ops.torchsparse_utils import sparse_tensor_summation
from torch_scatter import scatter
from misc.vis import vis_3D_in_occ_anchor_refmnt
from src.modules import AnchorSCNN


def conv1d_bn_relu(in_channels, out_channels, kernel_size=1, stride=1, padding=1, **kwargs):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False, **kwargs),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    )


class HypothesisDecoder(nn.Module):
    def __init__(self, in_dim=64, h_dim=128, kernel_size=3, padding=1):
        super(HypothesisDecoder, self).__init__()
        self.net = nn.Sequential(
            conv1d_bn_relu(in_dim, h_dim, kernel_size, 1, padding),
            conv1d_bn_relu(h_dim, h_dim, kernel_size, 1, padding),
            conv1d_bn_relu(h_dim, h_dim, kernel_size, 1, padding),
            nn.Conv1d(h_dim, 1, kernel_size, 1, padding)
        )
        self.sparse_interp = ME.MinkowskiInterpolation()

    def forward(self, x_sparse, pts_depthmap_hyp, pts_feat, pts_batch):
        """
        @var n_pts_per_depthmap: 56x56, # of 2d pts on the init depthmap prediction
        @var n_pts_all_views: b x n_views x n_pts_per_depthmap
        @param x_sparse: dict of sparse 3D geometric features,
        {
            'pts': (n, 3), pt coordinates
            'batch': (n,), the batch index for each 3d pt
            'sparse': Minkowski Sparse Tensor, shape of (n, c)
        }
        @param pts_depthmap_hyp: (n_pts_all_views, n_hyp==2h+1, 3), h (# hypothesis pts per depth pixel) == 3
        @param pts_feat: (n_pts_all_views, n_hyp, gru_fusion.channels_backbone2d[scale]),
            variance feature using the reprojection of hypothesis pts, applied per backbone2d channel
        @param pts_batch: (n_pts_all_views), batch idx for each hypothesis pt of all depthmap across n_views
        @var feats: (n_pts_all_views, n_hyp, c==x['sparse'].shape[1])
        @return preds: (n_pts_per_depthmap, n_hyp), for b x n_views depth pred offset
        """
        n_pts_all_views = pts_depthmap_hyp.shape[0]
        n_hyp = pts_depthmap_hyp.shape[1]

        # closet to origin pt [0,0,0] in each batch idx in x['batch']
        min_pts = scatter(x_sparse['pts'], x_sparse['batch'], dim=0, reduce='min')
        # how much each point offset from the min_pts in the current batch
        # will be used as an idx to access sparse 3d feature
        pts_idx = pts_depthmap_hyp - min_pts[pts_batch].unsqueeze(1).expand(*pts_depthmap_hyp.shape)

        # raw idx in minkowski tensor form
        pts_idx = (pts_idx / x_sparse['res']) * x_sparse['stride']

        pts_batch_unrolled = pts_batch.unsqueeze(1).repeat(1, n_hyp).unsqueeze(2)
        # get batch_idx cat with 3d pts, ME assumes the coord sequence as [b,x,y,z], shape of (n_pts_all_views, n_hyp, 4)
        pts_idx_batched = torch.cat((pts_batch_unrolled.float(), pts_idx), dim=2)
        pts_idx_batched_flat = pts_idx_batched.view(n_pts_all_views * n_hyp, 4)

        # sparse version of F.grid_sample()
        feats = self.sparse_interp(x_sparse['sparse'], pts_idx_batched_flat)
        feats = feats.view(n_pts_all_views, n_hyp, -1)

        # cat at dim=2 to create pts_feat vol as (n_pts_all_views, n_hyp, self.net.in_dim)
        pts_feat = feats if pts_feat is None else torch.cat((feats, pts_feat), dim=2)
        pts_feat = pts_feat.transpose(2, 1)  # (n_pts_all_views, self.net.in_dim, n_hyp)
        preds = F.softmax(self.net(pts_feat).squeeze(1), dim=1)  # (n_pts_all_views, n_hyp, 1) -> (n_pts_all_views, 1)
        return preds


class AnchorFeatureDecoder(nn.Module):
    def __init__(self, cfg, in_dim, out_dim, n_vox=24):
        super(AnchorFeatureDecoder, self).__init__()
        self.cfg = cfg
        self.net = AnchorSCNN(
            in_channels=in_dim,
            out_channels=out_dim,
            dropout=cfg.SPARSEREG.DROPOUT
        )
        self.n_vox = n_vox
        self.sparse_interp = ME.MinkowskiInterpolation()

    def forward(self, depth, img_size, rotmats, tvecs, K,
                scale=0, feat=None, up_coords=None,
                interval=None, origin=None,
                visualize_depth_backproj=False):
        """
        @param depth: (n_views, h, w), depth prediction
        @param img_size: (h, w), image size
        @param rotmats: (n_views, 3, 3), rotation matrix
        @param tvecs: (n_views, 3), translation vector
        @param K: (n_views, 3, 3), camera intrinsic
        @param scale: int, scale of the feature map
        @param feat: (n_views, c, h, w), sparse raw 3D feature from GRU
        @param up_coords: (n_views, 3, h, w), unsampled coords
        @param interval: float, interval of the voxel grid
        @param origin: (3,), origin of the voxel grid
        @param visualize_depth_backproj: bool, whether to visualize the back-projected depth
        """
        n_views = depth.shape[0]
        with torch.no_grad():
            # back-project the predicted depth points to world coordinates
            K_inv = torch.inverse(K[2:-2])
            R_T = rotmats[2:-2].transpose(2, 1)

            # flatten the depth idx to (n_views, 3, h*w), where each entry has the corresponding pixel idx as img_size
            # doing so is to bring in the correlation with dbatch.images, such that dbatch.K can be directly used
            pts_img_grid_homo_flat = geo.batched_build_img_pts_tensor(n_views, img_size, depth.shape[1:])
            pts_img_grid_homo_flat = pts_img_grid_homo_flat.type_as(depth)  # (n_views, 3, h*w), 3: [u,v,1]
            depth_flat = depth.view(n_views, 1, -1)  # (n_views, 1, h*w), 1: Z

            # from the projection equation, XYZ = func(u,v,1) * Z as back projection, with Z determined,
            # the back projection confirm XYZ as 3D pts
            pts_img_grid_depth = pts_img_grid_homo_flat * depth_flat
            pts_depth_bp_in_3d = torch.bmm(R_T,
                                           torch.bmm(K_inv, pts_img_grid_depth) -
                                           tvecs[2:-2].unsqueeze(-1))  # (n_views, 3, h*w), 3: [X,Y,Z]
            pts_depth_bp_in_3d = pts_depth_bp_in_3d.transpose(2, 1).reshape(-1, 3)

            # torch-geometric based voxelization: create anchor pts for depth pt cloud
            anchor_pts, anchor_idx3d = geo.voxelize(pts_depth_bp_in_3d,
                                                    0.04 * 2 ** scale,
                                                    origin=origin)

            # anchor_valid_vol = torch.zeros(self.n_vox, self.n_vox, self.n_vox, dtype=torch.bool).cuda()
            anchor_idx3d = torch.clamp(anchor_idx3d, 0, self.n_vox - 1)
            anchor_idx3d = torch.cat((torch.zeros(anchor_idx3d.shape[0], dtype=torch.uint8).unsqueeze(1).cuda(),
                                      anchor_idx3d), dim=1)

            if visualize_depth_backproj is True:
                print('VIS OCC ANCHOR PTS AND DEPTH PTS, scale {}'.format(scale))
                vis_3D_in_occ_anchor_refmnt(anchor_pts, pts_depth_bp_in_3d, scale=scale,
                                            up_coords=up_coords, tsdf_target=tsdf_target,
                                            origin=origin)

        voxel_center_to_edge = 0.04 * 2 ** scale / 2
        depth_bp_pts_pt_tensor = tst.PointTensor(torch.ones(pts_depth_bp_in_3d.shape[0], 1).cuda(),
                                                 torch.cat([pts_depth_bp_in_3d - origin + voxel_center_to_edge,
                                                            torch.zeros(pts_depth_bp_in_3d.shape[0], 1).cuda()],
                                                           dim=1))

        gru_feat_sparse = ME.SparseTensor(feat, (up_coords / interval).int())
        anchor_feat = self.sparse_interp(gru_feat_sparse, anchor_idx3d.float())
        anchor_feat_sparse = tst.SparseTensor(anchor_feat.float(), anchor_idx3d)
        gru_feat_sparse = tst.SparseTensor(feat.float(),
                                                (up_coords[:, [1, 2, 3, 0]] / interval).int())
        gru_feat_sparse = sparse_tensor_summation(gru_feat_sparse, anchor_feat_sparse)
        gru_feat_point = tst.PointTensor(feat.float(),
                                              (up_coords[:, [1, 2, 3, 0]] / interval))
        out_feat = self.net(depth_bp_pts_pt_tensor, gru_feat_sparse, gru_feat_point,
                                 pres=1, vres=self.cfg.VOXEL_SIZE * 2 ** scale)

        return out_feat
