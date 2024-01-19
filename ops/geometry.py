import numpy as np
import torch
import cv2
from torch_scatter import scatter
from torchvision import transforms
from torch_geometric.nn import voxel_grid


def voxelize(pts, edge_len, origin=None):
    """ drift depth back projection transform,
    to output correct anchor_pts so that occupancy can be correctly visualized.
    then voxelize.
    @param pts: (h*w*n_views, 3), 3d pts back-projected from n_views
    @param edge_len: voxel edge length
    @param origin: (3, ), origin of the original FBV, used to drift the depth bp pts
    @return:
    """
    bbox_min = pts.min(dim=0)[0]
    bbox_max = pts.max(dim=0)[0]
    grid_size = torch.ceil((bbox_max - bbox_min) / edge_len).long()

    # offset that used to drift current coord from origin (in pts) to current bbox
    # for anchor_idx3d, but anchor_pts remain coz it is constructed in-place here
    offset = ((bbox_min + edge_len / 2.) - origin).unsqueeze(0).to('cuda')
    offset = torch.round(offset / edge_len).int()

    # get 1d voxel idx of pts
    pts_voxel_idx = voxel_grid(pos=pts, batch=None, size=edge_len, start=bbox_min, end=bbox_max)

    # determine unique of 1d anchor indices w/ batch information
    anchor_idx, pts_inv_idx = torch.unique(pts_voxel_idx, return_inverse=True)

    # convert to 3d idx for sparse convs
    anchor_idx3d = torch.zeros((anchor_idx.shape[0], 3), dtype=torch.int, device=pts.device)  # x, y, z
    anchor_idx3d[:, 2] = torch.div(anchor_idx, (grid_size[0] * grid_size[1]), rounding_mode='trunc')
    anchor_idx3d[:, 1] = torch.div((anchor_idx - anchor_idx3d[:, 2] * (grid_size[0] * grid_size[1])),
                                   grid_size[0], rounding_mode='trunc')
    anchor_idx3d[:, 0] = (anchor_idx - anchor_idx3d[:, 2] * (grid_size[0] * grid_size[1])) % grid_size[0]
    anchor_pts = anchor_idx3d * edge_len + bbox_min + edge_len / 2.
    # anchor_idx3d -= anchor_idx3d.min(dim=0)[0]  # make min of anchor idx3d [0, 0, 0]
    anchor_idx3d += offset  # drift to match bbox, also match with gt tsdf vert

    return anchor_pts, anchor_idx3d


def build_img_pts_grid(img_size=(240, 320), plane_size=(56, 56)):
    """Return tensor of (3, plane_size[1]*plane_size[0]), 3 is xyz"""
    pts_x = np.linspace(0, img_size[1] - 1, plane_size[1], dtype=np.float32)
    pts_y = np.linspace(0, img_size[0] - 1, plane_size[0], dtype=np.float32)
    pts_xx, pts_yy = np.meshgrid(pts_x, pts_y)

    pts_xx = pts_xx.reshape(-1)  # flatten
    pts_yy = pts_yy.reshape(-1)
    z = np.ones(pts_xx.shape[0], dtype=np.float32)

    pts = np.stack((pts_xx, pts_yy, z))
    return pts


def batched_build_img_pts_tensor(n_view, img_size=(240, 320), plane_size=(60, 80)):
    """Building the grid of tensor, return tensor of (n_view, 3, plane_size[1]*plane_size[0])"""
    pts = torch.from_numpy(build_img_pts_grid(img_size, plane_size))
    pts = pts[None].repeat(n_view, 1, 1)
    return pts


def batched_build_plane_sweep_volume_tensor(R, t, K,
                                            depth_start=.5,
                                            depth_interval=.05,
                                            n_planes=96,
                                            img_size=(480, 640),
                                            plane_size=(60, 80)):
    """ Notice that the arg here should be matched with the predetermined setup here.
        At this point, just trying to infer depth from Neucon FPN+MNASNet 2d backbone.

        For each batch, or each fragment in Neucon, a cost volume will be init here.
        Default depth plane size (60, 80) to match with feat_eighth so that no oom.
        As experimented in 3dvnet, (56, 56) is also good.
    """
    n_view = R.shape[0]

    # build img pts grid in numpy, duplicate fronto-parallel plane into multi fp planes
    depth_end = depth_start + (n_planes - 1) * depth_interval
    pts_x = np.linspace(0, img_size[1] - 1, plane_size[1], dtype=np.float32)
    pts_y = np.linspace(0, img_size[0] - 1, plane_size[0], dtype=np.float32)
    z = np.linspace(depth_start, depth_end, n_planes, dtype=np.float32)

    # create 2d coord for the plane
    pts_xx, pts_yy = np.meshgrid(pts_x, pts_y)
    pts = np.stack((pts_xx, pts_yy))

    # init z grid, with n_planes wide
    pts = np.repeat(pts[:, None, :], z.shape[0], axis=1)

    # turn into homo coord
    pts = np.concatenate((pts, np.ones((1, z.shape[0], *plane_size))), axis=0)

    # turn 1d into 4d, create dim at 0, 2, 3 position. The product is Eq. 1 in MVSNet
    pts = pts * z[None, :, None, None]

    # plane sweep tensor, [b, 3, x]
    # flatten [3, n_planes, plane_size, plane_size] tensor into [3, x], and then take batch size as [b, 3, x]
    pts_tensor = torch.from_numpy(pts).float().view(3, -1).unsqueeze(0).repeat(n_view, 1, 1).type_as(R)

    # perform batched mm to convert each of fp planes from sole imaging plane into multiple world coords
    K_inv = torch.inverse(K)
    R_T = torch.transpose(R, 2, 1)

    # create 3d point from the 2d plane, it can be done with 4x4 proj mat in homo coord as in kwea video, but not faster
    pts_cam = torch.bmm(K_inv, pts_tensor)
    pts_world = torch.bmm(R_T, pts_cam - t.unsqueeze(-1))  # add one more dim at the last dim

    return pts_world


def random_gravitational_rotation_scannet():
    theta = np.random.uniform(-np.pi, np.pi)
    rotvec = np.array([0, 0, theta], dtype=np.float32)
    R, _ = cv2.Rodrigues(rotvec)
    R = R.astype(np.float32)
    return R
