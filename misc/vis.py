import torch
import numpy as np
import cv2
import open3d as o3d
import functools
from skimage import measure, transform


def vis_3D_in_occ_anchor_refmnt(anchor_pts, depth_pts, scale, origin, tsdf_target=None, up_coords=None):
    # anchor pts as occupancy box center, plot occupancy
    anchor_pts = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(anchor_pts.to('cpu').numpy()))
    anchor_pts.paint_uniform_color([1, 0, 0])
    anchor_occ_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(anchor_pts,
                                                                           voxel_size=.04 * 2 ** scale)

    # plot back-projected depth pts
    depth_pts = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(depth_pts.to('cpu').numpy()))
    depth_pts.paint_uniform_color([0, 0, 1])

    # just for gt tsdf vert comparison
    if tsdf_target is not None:
        dense_tsdf_target_vol = sparse_to_dense_torch(up_coords[:, 1:].to('cpu').long(),
                                                      tsdf_target.squeeze(-1).to('cpu'), [96, 96, 96],
                                                      default_val=1, device='cpu')
        verts_gt, faces_gt, norms_gt, _ = measure.marching_cubes(dense_tsdf_target_vol.data.numpy(), level=0)
        verts_gt = verts_gt * 0.04 + origin.to('cpu').numpy()
        pcd_gt_tsdf = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(verts_gt))
        pcd_gt_tsdf.paint_uniform_color([0, 1, 0])

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
    list_to_vis = [anchor_pts, anchor_occ_voxel_grid, depth_pts, pcd_gt_tsdf, axes] if tsdf_target is not None else [
        anchor_pts, anchor_occ_voxel_grid, depth_pts, axes]
    visualize_ptcloud_mesh(list_to_vis)
