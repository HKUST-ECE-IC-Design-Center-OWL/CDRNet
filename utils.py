import os
import torch
import torch_geometric
import trimesh
import numpy as np
from skimage import measure, transform
from loguru import logger
from tools.render import Visualizer
import cv2
import open3d as o3d

VALID_CLASS_IDS_NYU40 = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])  # 20 classes out of nyu40id


# torch.no_grad warpper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        if len(vars.shape) == 0:
            return vars.data.item()
        else:
            return [v.data.item() for v in vars]
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, (torch.Tensor, torch_geometric.data.batch.Batch)):
        return vars.cuda(non_blocking=True)
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tocuda".format(type(vars)))


def coordinates(voxel_dim, device=torch.device('cuda')):
    """ 3d meshgrid of given size.

    Args:
        voxel_dim: tuple of 3 ints (nx,ny,nz) specifying the size of the volume

    Returns:
        torch long tensor of size (3,nx*ny*nz)
    """

    nx, ny, nz = voxel_dim
    x = torch.arange(nx, dtype=torch.long, device=device)
    y = torch.arange(ny, dtype=torch.long, device=device)
    z = torch.arange(nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z)
    return torch.stack((x.flatten(), y.flatten(), z.flatten()))


def apply_log_transform(tsdf):
    sgn = torch.sign(tsdf)
    out = torch.log(torch.abs(tsdf) + 1)
    out = sgn * out
    return out


def sparse_to_dense_torch_batch(locs, values, dim, default_val):
    dense = torch.full([dim[0], dim[1], dim[2], dim[3]], float(default_val), device=locs.device)
    dense[locs[:, 0], locs[:, 1], locs[:, 2], locs[:, 3]] = values
    return dense


def sparse_to_dense_torch(locs, values, dim, default_val, device):
    dense = torch.full([dim[0], dim[1], dim[2]], float(default_val), device=device)
    if locs.shape[0] > 0:
        dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense


def sparse_to_dense_channel(locs, values, dim, c, default_val, device):
    dense = torch.full([dim[0], dim[1], dim[2], c], float(default_val), device=device)
    if locs.shape[0] > 0:
        dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense


def sparse_to_dense_np(locs, values, dim, default_val):
    dense = np.zeros([dim[0], dim[1], dim[2]], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense


class SaveScene(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.log_dir = os.path.join(cfg.LOGDIR, cfg.MODE, 'scene_' + cfg.DATASET)
        logger.info('saving inference scenes (and transferred) into log dir {}'.format(self.log_dir))

        self.scene_name = None
        self.global_origin = None
        self.coords = None
        self.keyframe_id = None
        if cfg.VIS_INCREMENTAL:
            self.vis = Visualizer()

        # export semseg transferred mesh for one scene you specify
        self.transferred_mesh_save = (cfg.MODE != 'test')
        self.assigned_scene_name = None

    def close(self):
        self.vis.close()
        cv2.destroyAllWindows()

    def reset(self):
        self.keyframe_id = 0

    def tsdf_semseg_2colormesh(self, voxel_size, origin, tsdf_vol, semseg_vol, save_transfer_path, scene_name,
                               trgt_tsdf_path,
                               cmap='nyu40'):
        verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0)

        # turn semseg from one-hot to int, in atlas postprocess()
        if semseg_vol.ndim == 4:
            semseg_vol = semseg_vol.argmax(3)  # now ndim=

        # notice here has to be voxel grid coord
        verts_ind = np.round(verts).astype(int)
        verts = verts * voxel_size + origin  # voxel grid coordinates to world coordinates
        vertex_attributes = {}
        semseg = semseg_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        vertex_attributes['semseg'] = semseg.astype(np.int32)

        # color map
        if cmap == 'nyu40':
            cmap = np.array(NYU40_COLORMAP)
        else:
            raise NotImplementedError('colormap %s' % cmap)
        label_viz = semseg.copy()
        label_viz[(label_viz < 0) | (label_viz >= len(cmap))] = 0

        print('label viz: {}'.format(label_viz))
        print('label viz max: {}'.format(np.max(label_viz)))
        print('label viz unique: {}'.format(np.unique(label_viz)))

        colors = cmap[label_viz, :]

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms,
                               vertex_colors=colors, vertex_attributes=vertex_attributes,
                               process=False)

        # save semseg attribute for further evaluation on miou
        if trgt_tsdf_path is not None:
            mesh_trgt = trimesh.load(trgt_tsdf_path, process=False)
            mesh_transfer_semseg_vec = self.project_to_trgt_mesh_return_semseg(mesh, mesh_trgt, 'semseg',
                                                                               transferred_mesh_save=self.transferred_mesh_save)
            if self.transferred_mesh_save:
                mesh_transfer_semseg_vec[1].export(os.path.join(save_transfer_path, '%s_transfer.ply' % scene_name))
                np.savetxt(os.path.join(save_transfer_path, '%s.txt' % scene_name), mesh_transfer_semseg_vec[0],
                           fmt='%d')
            else:
                np.savetxt(os.path.join(save_transfer_path, '%s.txt' % scene_name), mesh_transfer_semseg_vec, fmt='%d')
        mesh.export(os.path.join(save_transfer_path, '{}.ply'.format(self.scene_name)))

        return mesh

    @staticmethod
    def project_to_trgt_mesh_return_semseg(from_mesh, to_mesh, attribute, dist_thresh=None,
                                           transferred_mesh_save=False):
        if len(from_mesh.vertices) == 0:
            to_mesh.vertex_attributes[attribute] = np.zeros((0), dtype=np.uint8)
            to_mesh.visual.vertex_colors = np.zeros((0), dtype=np.uint8)
            return to_mesh

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(from_mesh.vertices)
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        pred_ids = from_mesh.vertex_attributes[attribute]
        pred_colors = from_mesh.visual.vertex_colors

        matched_ids = np.zeros((to_mesh.vertices.shape[0]), dtype=np.uint8)
        matched_colors = np.zeros((to_mesh.vertices.shape[0], 4), dtype=np.uint8)

        for i, vert in enumerate(to_mesh.vertices):
            _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
            if dist_thresh is None or dist[0] < dist_thresh:
                matched_ids[i] = pred_ids[inds[0]]
                matched_colors[i] = pred_colors[inds[0]]

        # to dump transferred mesh
        if transferred_mesh_save:
            mesh = to_mesh.copy()
            mesh.vertex_attributes[attribute] = matched_ids
            mesh.visual.vertex_colors = matched_colors
            return matched_ids, mesh
        return matched_ids

    def vis_incremental(self, batch_idx, imgs, outputs):
        tsdf_volume = outputs['scene_tsdf'][batch_idx].data.cpu().numpy()
        origin = outputs['origin'][batch_idx].data.cpu().numpy()
        if self.cfg.DATASET == 'demo':
            origin[2] -= 1.5

        if (tsdf_volume == 1).all():
            logger.warning('No valid partial data for scene {}'.format(self.scene_name))
        else:
            # Marching cubes
            mesh = self.tsdf_semseg_2colormesh(self.cfg.MODEL.VOXEL_SIZE, origin, tsdf_volume)
            # vis
            key_frames = []
            for img in imgs[::3]:
                img = img.permute(1, 2, 0)
                img = img[:, :, [2, 1, 0]]  # adjust the sequence of RGB
                img = img.data.cpu().numpy()
                img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
                key_frames.append(img)
            # print('key shape is {}'.format(len(key_frames)))  # shows that it is first 3 imported
            key_frames = np.concatenate(key_frames, axis=0)
            cv2.imshow('Selected Keyframes', key_frames / 255)
            cv2.waitKey(1)
            # vis mesh
            self.vis.vis_mesh(mesh)

    def save_incremental(self, epoch_idx, batch_idx, outputs):
        save_path = os.path.join('incremental_' + self.log_dir + '_' + str(epoch_idx), self.scene_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        tsdf_volume = outputs['scene_tsdf'][batch_idx].data.cpu().numpy()
        origin = outputs['origin'][batch_idx].data.cpu().numpy()
        if self.cfg.DATASET == 'demo':
            origin[2] -= 1.5

        if (tsdf_volume == 1).all():
            logger.warning('No valid partial data for scene {}'.format(self.scene_name))
        else:
            # Marching cubes
            mesh = self.tsdf2mesh(self.cfg.MODEL.VOXEL_SIZE, origin, tsdf_volume)
            # save
            mesh.export(os.path.join(save_path, 'mesh_{}.ply'.format(self.keyframe_id)))

    def save_scene_eval(self, epoch, outputs, batch_idx=0, assign_name=None):
        tsdf_volume = outputs['scene_tsdf'][batch_idx].data.cpu().numpy()
        if self.cfg.VIS_MESH_SEMSEG:
            semseg_volume = outputs['scene_semseg'][batch_idx].data.cpu().numpy()
        origin = outputs['origin'][batch_idx].data.cpu().numpy()

        if (tsdf_volume == 1).all():
            logger.warning('No valid data for scene {}'.format(self.scene_name))
        else:
            # save tsdf volume for atlas evaluation
            if self.cfg.VIS_MESH_SEMSEG:
                data = {'origin': origin,
                        'voxel_size': self.cfg.MODEL.VOXEL_SIZE,
                        'tsdf': tsdf_volume,
                        'semseg': semseg_volume}
            else:
                data = {'origin': origin,
                        'voxel_size': self.cfg.MODEL.VOXEL_SIZE,
                        'tsdf': tsdf_volume}
            save_path = '{}_fusion_eval_{}'.format(self.log_dir, epoch)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.savez_compressed(
                os.path.join(save_path, '{}.npz'.format(self.scene_name)),
                **data)

            if self.cfg.MODE in ['val', 'test']:
                trgt_tsdf_path = os.path.join('/media/zhongad/2TB/dataset/scannet/scans/', self.scene_name,
                                              self.scene_name + '_vh_clean_2.ply')
            else:
                trgt_tsdf_path = None

            if self.cfg.VIS_MESH_SEMSEG:
                mesh = self.tsdf_semseg_2colormesh(self.cfg.MODEL.VOXEL_SIZE, origin, tsdf_volume, semseg_volume,
                                                   save_transfer_path=save_path, scene_name=self.scene_name,
                                                   trgt_tsdf_path=trgt_tsdf_path)
            else:
                print('voxel size is {}'.format(self.cfg.MODEL.VOXEL_SIZE))
                mesh = self.tsdf2mesh(self.cfg.MODEL.VOXEL_SIZE, origin, tsdf_volume)

            if assign_name is not None:
                self.assigned_scene_name = assign_name

            if self.assigned_scene_name is not None:
                mesh.export('{}.ply'.format(self.assigned_scene_name))
            else:
                mesh.export(os.path.join(save_path, '{}.ply'.format(self.scene_name)))

    def __call__(self, outputs, inputs, epoch_idx, assign_name=None):
        # only reach scene's frag end, fuse_to_global will create output['scene_name']
        # no scene saved, skip
        if "scene_name" not in outputs.keys():
            return

        if assign_name is not None:
            self.assigned_scene_name = assign_name
        batch_size = len(outputs['scene_name'])
        for i in range(batch_size):
            scene = outputs['scene_name'][i]
            self.scene_name = scene.replace('/', '-')

            if self.cfg.SAVE_SCENE_MESH:
                self.save_scene_eval(epoch_idx, outputs, i)


def depth_val_proc(depth_frame):
    # depth_frame = depth_frame.astype(np.float32)
    depth = (np.clip((depth_frame - .5) / 5, 0, 1) * 255).astype(np.uint8)
    depth = cv2.applyColorMap(depth,
                              cv2.COLORMAP_JET)  # DO NOT modify the depth after apply color map except for modify as 0, otherwise raise bug: messed rgb val
    depth[depth_frame == 0] = 0
    return depth


def vis_2D_depth_prediction_comparison(depths_pred_raw, depths_pred_offseted, gt_depths, depth_from_mesh=None):
    """2D VIS: The just-upsampled (raw) depth vs. pointflow-offseted depth"""
    depths_pred_raw = depths_pred_raw.detach().cpu().numpy()
    depths_pred_offseted = depths_pred_offseted.detach().cpu().numpy()
    gt_depths = gt_depths.detach().cpu().numpy()
    depth_from_mesh = depth_from_mesh.detach().cpu().numpy()
    for i in range(depths_pred_raw.shape[0]):
        depth_pred_raw = depth_val_proc(depths_pred_raw[i])
        depth_pred_offseted = depth_val_proc(depths_pred_offseted[i])
        gt_depth = depth_val_proc(transform.resize(gt_depths[i], depths_pred_raw[i].shape))
        viz = np.hstack((depth_pred_raw, depth_pred_offseted, gt_depth))
        if depth_from_mesh is not None:
            viz = np.hstack((viz, depth_val_proc(depth_from_mesh[i])))
            cv2.imshow('left:before; mid:after; gt; depth_from_mesh', viz)
        else:
            cv2.imshow('[occrefmnt/gt] left:before;mid:after;gt', viz)
        # print('vis depth {}'.format(i))
        cv2.waitKey(200)


NYU40_COLORMAP = [
    (0, 0, 0),
    (174, 199, 232),  # wall
    (152, 223, 138),  # floor
    (31, 119, 180),  # cabinet
    (255, 187, 120),  # bed
    (188, 189, 34),  # chair
    (140, 86, 75),  # sofa
    (255, 152, 150),  # table
    (214, 39, 40),  # door
    (197, 176, 213),  # window
    (148, 103, 189),  # bookshelf
    (196, 156, 148),  # picture
    (23, 190, 207),  # counter
    (178, 76, 76),
    (247, 182, 210),  # desk
    (66, 188, 102),
    (219, 219, 141),  # curtain
    (140, 57, 197),
    (202, 185, 52),
    (51, 176, 203),
    (200, 54, 131),
    (92, 193, 61),
    (78, 71, 183),
    (172, 114, 82),
    (255, 127, 14),  # refrigerator
    (91, 163, 138),
    (153, 98, 156),
    (140, 153, 101),
    (158, 218, 229),  # shower curtain
    (100, 125, 154),
    (178, 127, 135),
    (120, 185, 128),
    (146, 111, 194),
    (44, 160, 44),  # toilet
    (112, 128, 144),  # sink
    (96, 207, 209),
    (227, 119, 194),  # bathtub
    (213, 92, 176),
    (94, 106, 211),
    (82, 84, 163),  # otherfurn
    (100, 85, 144)
]
