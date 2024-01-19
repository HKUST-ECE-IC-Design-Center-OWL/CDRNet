import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
import MinkowskiEngine as ME

from torchsparse.tensor import PointTensor
from torchsparse.utils import *
from ops.torchsparse_utils import *
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

__all__ = ['SPVCNN', 'SConv3d', 'ConvGRU', 'MinkUNet18ADecoder']


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride), spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True), spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride), spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=1), spnn.BatchNorm(outc))

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class SPVCNN(nn.Module):
    """Sparse 3D backbone"""

    def __init__(self, **kwargs):
        super().__init__()

        self.dropout = kwargs['dropout']

        cr = kwargs.get('cr', 1.0)
        cs = [32, 64, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.stem = nn.Sequential(
            spnn.Conv3d(kwargs['in_channels'], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),  # f: 2x2x2 3D conv kernel
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),  # f: 3x3x3 multi-residual conv
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[2], cs[3], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[3] + cs[1], cs[3], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[3], cs[4], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[4] + cs[0], cs[4], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
            )
        ])

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[2]),
                nn.BatchNorm1d(cs[2]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[2], cs[4]),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(True),
            )
        ])

        self.weight_initialization()

        if self.dropout:
            self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        # x: SparseTensor z: PointTensor
        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        z1 = voxel_to_point(x2, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y3 = point_to_voxel(x2, z1)
        if self.dropout:
            y3.F = self.dropout(y3.F)
        y3 = self.up1[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up1[1](y3)

        y4 = self.up2[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up2[1](y4)
        z3 = voxel_to_point(y4, z1)
        z3.F = z3.F + self.point_transforms[1](z1.F)

        return z3.F


class SConv3d(nn.Module):
    """Sparse Conv in GRU"""

    def __init__(self, inc, outc, pres, vres, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = spnn.Conv3d(inc,
                               outc,
                               kernel_size=ks,
                               dilation=dilation,
                               stride=stride)
        self.point_transforms = nn.Sequential(
            nn.Linear(inc, outc),
        )
        self.pres = pres  # pt resolution
        self.vres = vres  # voxel resolution

    def forward(self, z):
        # x: SparseTensor z: PointTensor
        x = initial_voxelize(z, self.pres, self.vres)
        x = self.net(x)
        out = voxel_to_point(x, z, nearest=False)  # pt tensor
        out.F = out.F + self.point_transforms(z.F)
        return out


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128, pres=1, vres=1):
        super(ConvGRU, self).__init__()
        self.convz = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        self.convr = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        self.convq = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)

    def forward(self, h, x):
        """

        :param h: PointTensor, hidden state
        :param x: PointTensor, 3D geometric features
        :return: h.F: Tensor (N, C)
        """
        hx = PointTensor(torch.cat([h.F, x.F], dim=1), h.C)

        z = torch.sigmoid(self.convz(hx).F)
        r = torch.sigmoid(self.convr(hx).F)
        x.F = torch.cat([r * h.F, x.F], dim=1)
        q = torch.tanh(self.convq(x).F)

        h.F = (1 - z) * h.F + z * q
        return h.F


class AnchorSCNN(nn.Module):
    """SCNN for anchor occupancy"""

    def __init__(self, **kwargs):
        super().__init__()

        self.dropout = kwargs['dropout']

        cr = kwargs.get('cr', 1.0)
        cs = [96, 128, 48]
        cs = [int(cr * x) for x in cs]

        self.stem_depth_bp_occ = nn.Sequential(
            spnn.Conv3d(1, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True)
        )
        self.stem_feat = nn.Sequential(
            spnn.Conv3d(kwargs['in_channels'], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True)
        )

        self.stage = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
        )

        self.up = nn.ModuleList([
            BasicDeconvolutionBlock(cs[1], cs[2], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[2] + cs[0], cs[2], ks=3, stride=1, dilation=1),
            )
        ])

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[1]),
                nn.BatchNorm1d(cs[1]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[1], cs[2]),
                nn.BatchNorm1d(cs[2]),
                nn.ReLU(True),
            )
        ])

        self.occupancy_option = nn.Linear(cs[2], kwargs['out_channels'])

        self.weight_initialization()

        if self.dropout:
            self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, z_depth, x_feat_raw, z_feat_raw, pres, vres):
        # x: SparseTensor z: PointTensor
        x0 = initial_voxelize(z_depth, pres, vres)  # after this fn, z_depth updated as voxel_coord
        x0 = self.stem_depth_bp_occ(x0)
        x_feat_raw = self.stem_feat(x_feat_raw)
        x0 = sparse_tensor_summation(x0, x_feat_raw)

        z0 = voxel_to_point(x0, z_feat_raw, nearest=False)  # depth voxelization refinement
        z0.F = z0.F
        x1 = point_to_voxel(x0, z0)
        x1 = self.stage(x1)
        z1 = voxel_to_point(x1, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y2 = point_to_voxel(x1, z1)
        if self.dropout:
            y2.F = self.dropout(y2.F)
        y2 = self.up[0](y2)
        y2 = torchsparse.cat([y2, x0])
        y2 = self.up[1](y2)
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        out = self.occupancy_option(z2.F)

        return out


if __name__ == '__main__':
    from torchsparse.tensor import PointTensor
    from torchsparse.utils.quantize import sparse_quantize
    import torchsparse
    import numpy as np

    alpha = 1
    voxel_size = .04
    n_stage = 3
    dropout = False
    ch_in = [80 * alpha + 1, 96 + 40 * alpha + 2 + 1, 48 + 24 * alpha + 2 + 1, 24 + 24 + 2 + 1]
    sp_convs = nn.ModuleList()
    for i in range(3):
        sp_convs.append(
            SPVCNN(in_channels=ch_in[i], pres=1, cr=1 / 2 ** i, vres=voxel_size * 2 ** (n_stage - 1 - i),
                   dropout=dropout)
        )
    # x = torch.rand(1, a, b, c)
    np.random.seed(seed=0)
    inputs = np.random.uniform(-100, 100, size=(13824, 81))
    feat, pcs = inputs, inputs[:, :4]
    pcs -= np.min(pcs, axis=0, keepdims=True)
    # these are for sparse tensor (voxelized), not point tensor
    # pcs, indices = sparse_quantize(pcs, voxel_size, return_index=True)
    # coords = np.zeros((pcs.shape[0], 4))
    # coords[:, :3] = pcs[:, :3]
    # coords[:, -1] = 0
    # coords = torch.as_tensor(coords, dtype=torch.int)
    # feats = torch.as_tensor(feat[indices], dtype=torch.float)

    pcs = torch.as_tensor(pcs, dtype=torch.int)
    feats = torch.as_tensor(feat, dtype=torch.float)
    x = PointTensor(feats, pcs)

    # out = model(x)
    # from torchsummaryX import summary
    # _ = summary(model=sp_convs[0], x=x)
    from thop import profile, clever_format
    from tools.count.thop_count import count_sparseConv

    print(sp_convs[0])
    macs, params = profile(sp_convs[0], inputs=(x,),
                           custom_ops={torchsparse.nn.Conv3d: count_sparseConv})
    macs, params = clever_format([macs, params], '%.3f')
    from loguru import logger

    logger.info('macs: {}, params: {}'.format(macs, params))
    logger.info('******************* here ends thop')

    print('finish')
