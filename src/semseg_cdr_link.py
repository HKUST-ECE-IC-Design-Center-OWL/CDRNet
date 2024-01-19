import torch
from torch import nn
import MinkowskiEngine as ME
from MinkowskiEngine.utils.coords import get_coords_map
from MinkowskiEngine.MinkowskiCoordinateManager import CoordinateManager
import pdb
from loguru import logger


class SemsegCDRLink(nn.Module):
    def __init__(self, feat2d_channel, feat3d_dim, view_num):
        super(SemsegCDRLink, self).__init__()
        self.view_num = view_num
        self.feat2d_cnl = feat2d_channel

        self.view_fusion = nn.Sequential(
            ME.MinkowskiConvolution(feat2d_channel * view_num, feat2d_channel, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(feat2d_channel),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(feat2d_channel, feat3d_dim, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(feat3d_dim),
            ME.MinkowskiReLU(inplace=True)
        )

        self.fuse_to_3d = nn.Sequential(
            ME.MinkowskiConvolution(feat3d_dim * 2, feat3d_dim, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(feat3d_dim),
            ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, feat_2d_all, sparse_feat_3d, links, init_3d=None, scale=2, iter=None):
        """
        @param feat_2d_all: V_B * C * H * W, the raw 2d feat
        @param sparse_feat_3d:
        @param links: (n, 4, V), where 4: [batch_idx, v, u, valid]
        @param init_3d: (n, 3) Mink.SparseTensor feature; n = #mesh_vertex in fbv, 3 rgb feature
        @return:
        """
        n_views, bs, c, h, w = feat_2d_all.shape
        feat_2d_all = feat_2d_all.view(self.view_num, -1, c, h, w)  # View * B * C * H * W
        coords_map_in, coords_map_out = get_coords_map(init_3d, sparse_feat_3d)
        current_links = torch.zeros([sparse_feat_3d.shape[0], links.shape[1], links.shape[2]], dtype=torch.long).cuda()
        # current links are scaled from global links, same shape as feat_3d now
        current_links[coords_map_out, :] = links[coords_map_in, :]

        feat_2d_to_3d = torch.zeros([sparse_feat_3d.F.shape[0], self.view_num * self.feat2d_cnl], dtype=torch.float).cuda()
        for v in range(self.view_num):
            try:
                view_feat = feat_2d_all[v, current_links[:, 0, v], :, current_links[:, 1, v], current_links[:, 2, v]]
                # only when valid, view_feat will be non zero
                view_feat *= current_links[:, 3, v].unsqueeze(dim=1).float()
                feat_2d_to_3d[:, v * self.feat2d_cnl:(v + 1) * self.feat2d_cnl] = view_feat  # B * C
            except:
                logger.warning(
                    'no coord matching between init_3d and feat_3d at iter {}; scale {}; links.sum()={}, '
                    'links.shape={}' .format(iter, scale, links[:, -1, :].sum(), links.shape))

        feat_2d_to_3d = ME.SparseTensor(feat_2d_to_3d, sparse_feat_3d.C)  # with 3D coords, thus takes 3D conv in view_fusion
        feat_2d_to_3d = self.view_fusion(feat_2d_to_3d)
        sparse_feat_3d._F = torch.cat([sparse_feat_3d._F, feat_2d_to_3d._F], dim=-1)  # channel-wise
        fused_3d = self.fuse_to_3d(sparse_feat_3d)

        return fused_3d
