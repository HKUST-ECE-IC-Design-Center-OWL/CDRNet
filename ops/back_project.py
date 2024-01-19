import torch
from torch.nn.functional import grid_sample


def back_project(coords, origin, voxel_size, feats, KRcam):
    """
    Back project the image features to form a 3D (sparse) feature volume.
    1. Projection from the real space volume grid for n_view
    2. Create the belows for each view
        * mask according to the projected pixel position and image h/w
        * and 2D img correspondent to each voxel in the grid, im_grid
    3. Sampling (interpolation) on the grid to create 3D features, using
        * 2D features: feats_batch
        * 2D pixels projected from 3D grid voxel: im_grid, to ensured 3D features created in the FBV
    4. Aggregate multiple views features, then concat the 1-channel normalized depth with the 3D features

    @param coords: coordinates of voxels,
        dim: (# voxels, 4) (4 : batch ind, x, y, z)
    @param origin: origin of the partial voxel volume (xyz position of voxel (0, 0, 0))
        dim: (batch size, 3) (3: x, y, z)
    @param voxel_size: floats specifying the size of a voxel
    @param feats: image features
        dim: (# views, batch size, C, H, W)
    @param KRcam: projection matrix
        dim: (# views, batch size, 4, 4)

    @return: feature_volume_all: 3D feature volumes
        dim: (# voxels, c + 1)
    @return: count: number of times each voxel can be seen, out of n_view
        dim: (# voxels,)
    """
    n_views, bs, c, h, w = feats.shape

    feature_volume_all = torch.zeros(coords.shape[0], c + 1).cuda()
    count = torch.zeros(coords.shape[0]).cuda()

    for batch in range(bs):
        batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)
        # setup uniformly for each batch, to assign grid_batch according to origin_batch
        coords_batch = coords[batch_ind][:, 1:]

        coords_batch = coords_batch.view(-1, 3)
        origin_batch = origin[batch].unsqueeze(0)
        feats_batch = feats[:, batch]
        proj_batch = KRcam[:, batch]

        grid_batch = coords_batch * voxel_size + origin_batch.float()  # in mm
        # real space grid in mm
        rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1)  # duplicate n_views times and put it in a new dimension
        rs_grid = rs_grid.permute(0, 2, 1).contiguous()
        nV = rs_grid.shape[-1]
        # make it 3d homo pt
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).cuda()], dim=1)

        # 1. create the pixel-wise grid for each view from real space grid
        im_proj = proj_batch @ rs_grid  # for each pixel-voxel proj pair: 4x1 = 4x4 * 4x1, already in homo coord
        im_x, im_y, im_z = im_proj[:, 0], im_proj[:, 1], im_proj[:, 2]
        # change 2d homo coord into 2d cartesian coord by unifying the z axis of 2d homo coord
        im_x = im_x / im_z
        im_y = im_y / im_z

        # 2. im_grid is the receptive field for each view, used to sample feats_batch in rs_grid
        # normalize the pixels/horw into (0,1): (0,1) -> (0,2) -> (-1,1) for grid_sample()
        im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)
        mask = im_grid.abs() <= 1
        # valid only when both x and y abs smaller than 1, and z is in front of cam center
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

        feats_batch = feats_batch.view(n_views, c, h, w)
        im_grid = im_grid.view(n_views, 1, -1, 2)  # [n_view, 1, dx*dy*dz, 2], each voxel's normalized img grid in 2D
        # 3. grid_sample() requires input arg to be a grid whose range of (-1,1)
        # grid sample expects coords in (-1,1)
        # the arg "grid" serves as the index for input
        # doing sampling among the input H,W dim
        features = grid_sample(input=feats_batch, grid=im_grid, padding_mode='zeros', align_corners=True)

        features = features.view(n_views, c, -1)  # [n_view, c, dx*dy*dz]
        mask = mask.view(n_views, -1)
        im_z = im_z.view(n_views, -1)
        # remove nan
        features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
        im_z[mask == False] = 0

        count[batch_ind] = mask.sum(dim=0).float()

        # 4. aggregate multi view with current FBV
        features = features.sum(dim=0)
        mask = mask.sum(dim=0)
        invalid_mask = mask == 0
        # avoid div by 0, the invalid part of mask assign 1
        # in fact mask == 0 means that either im_x/w or im_y/h out of (0,1)
        mask[invalid_mask] = 1
        in_scope_mask = mask.unsqueeze(0)
        features /= in_scope_mask  # averaging features
        features = features.permute(1, 0).contiguous()

        # concat normalized depth value as one more channel in output 3d feature
        # won't affect the ultimate shape in each level, but good for tsdf learning
        # im_z now is with mask so that already bounded into FBV
        # sum across n_views, so that can be added as one more channel to the grid features
        im_z = im_z.sum(dim=0).unsqueeze(1) / in_scope_mask.permute(1, 0).contiguous()
        im_z_mean = im_z[im_z > 0].mean()
        im_z_std = torch.norm(im_z[im_z > 0] - im_z_mean) + 1e-5
        im_z_norm = (im_z - im_z_mean) / im_z_std
        im_z_norm[im_z <= 0] = 0
        features = torch.cat([features, im_z_norm], dim=1)

        feature_volume_all[batch_ind] = features
    return feature_volume_all, count
