"""
Derived from:
https://github.com/mit-han-lab/spvnas/blob/b24f50379ed888d3a0e784508a809d4e92e820c0/core/models/utils.py
"""
import torch
import torchsparse.nn.functional as F
from torchsparse import PointTensor, SparseTensor
from torchsparse.nn.utils import get_kernel_offsets
from torch_scatter import scatter

__all__ = ['initial_voxelize', 'point_to_voxel', 'voxel_to_point', 'sparse_tensor_summation']


# z: PointTensor
# return: SparseTensor
def initial_voxelize(z, init_res, after_res):
    new_float_coord = torch.cat(
        [(z.C[:, :3] * init_res) / after_res, z.C[:, -1].view(-1, 1)], 1)

    pc_hash = F.sphash(torch.floor(new_float_coord).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = F.sphashquery(pc_hash, sparse_hash)
    counts = F.spcount(idx_query.int(), len(sparse_hash))

    inserted_coords = F.spvoxelize(torch.floor(new_float_coord), idx_query,
                                   counts)
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = F.spvoxelize(z.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if z.additional_features is None or z.additional_features.get('idx_query') is None \
            or z.additional_features['idx_query'].get(x.s) is None:
        # pc_hash = hash_gpu(torch.floor(z.C).int())
        pc_hash = F.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = F.sphash(x.C)
        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = F.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if z.idx_query is None or z.weights is None or z.idx_query.get(
            x.s) is None or z.weights.get(x.s) is None:
        off = get_kernel_offsets(2, x.s, 1, device=z.F.device)
        # old_hash = kernel_hash_gpu(torch.floor(z.C).int(), off)
        old_hash = F.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)
        pc_hash = F.sphash(x.C.to(z.F.device))
        idx_query = F.sphashquery(old_hash, pc_hash)
        weights = F.calc_ti_weights(z.C, idx_query,
                                    scale=x.s[0]).transpose(0, 1).contiguous()
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_feat = F.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        new_feat = F.spdevoxelize(x.F, z.idx_query.get(x.s),
                                  z.weights.get(x.s))
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features

    return new_tensor


def sparse_tensor_summation(x0, x1):
    # output_coords = ref.C
    # x0 = point_to_voxel(x0, ref)
    # F.sphash then idx_query = F.sphashquery to find out the correct coord to create the output tensor
    #  see https://github.com/mit-han-lab/torchsparse/issues/121 & https://github.com/mit-han-lab/torchsparse/issues/112
    coord_cat = torch.cat((x0.C, x1.C))
    feat_cat = torch.cat((x0.F, x1.F))
    spt_hash = F.sphash(coord_cat)
    spt_hash_unique = torch.unique(spt_hash)
    idx_query = F.sphashquery(spt_hash, spt_hash_unique)
    counts = F.spcount(idx_query.int(), len(spt_hash_unique))

    new_coords = F.spvoxelize(coord_cat.float(), idx_query, counts).int()
    new_feat = scatter(src=feat_cat, index=idx_query, dim=0, reduce='sum')
    out_sparse_tensor = SparseTensor(new_feat, new_coords, x0.s)
    out_sparse_tensor.cmaps.setdefault(out_sparse_tensor.stride, out_sparse_tensor.coords)
    return out_sparse_tensor
