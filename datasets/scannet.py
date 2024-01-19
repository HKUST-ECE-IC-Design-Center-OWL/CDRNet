# This file is derived from [NeuralRecon](https://github.com/zju3dv/NeuralRecon) by Yiming Xie and Jiaming Sun.
# Modified by HaFred

""" Semseg CDR with matching matrix, an optimal version for reading matching matrix which is pre-constructed off runtime."""

import os
import numpy as np
import pickle
import cv2
import torch
import json
import random
import torch.nn.functional as F
import datasets.transforms as transforms
import imageio

from PIL import Image
from torch.utils.data import Dataset
from torch_geometric import data
from datasets.batch import Batch
from utils import VALID_CLASS_IDS_NYU40


# since the pkl is generated under pklpath, but not in the 2TB datapath, thus the following init input params are changed
def build_list_onset(frag_path):
    # load the train/test/val fragments and assign scenes' fragments
    with open(frag_path, 'rb') as f:
        metas = pickle.load(f)
    return metas


def read_cam_file(filepath, vid):
    intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsic', 'intrinsic_color.txt'), delimiter=' ')[:3, :3]
    intrinsics = intrinsics.astype(np.float32)
    poses = np.loadtxt(os.path.join(filepath, 'pose', '{}.txt'.format(str(vid))))
    return intrinsics, poses


def read_img(filepath):
    img = Image.open(filepath)
    return img


def read_depth(filepath):
    depth_im = cv2.imread(filepath, -1).astype(
        np.float32)
    depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    depth_im[depth_im > 3.0] = 0
    return depth_im


def read_label(filepath):
    label_im = Image.open(filepath)
    return label_im


class ScanNetDataset(Dataset):
    def __init__(self, datapath, mode, transforms, nviews, n_scales,
                 pklpath=None, fragpath=None,
                 n_src_on_either_side=2, crop=False,
                 scale_rgb=255., mean_rgb=[.485, .456, .406], std_rgb=[.485, .456, .406],
                 semseg_matching=False, depth_prediction=False,
                 datasplit=None,
                 nyu40_20_conversion=False):
        """
        Read in semseg link under data/scannet/matching_mat

        Notice that for now, dbatch data is located in /media/zhongad/2TB/dataset/scannet/3dvnet_proc, while fragments are in data/scannet.
        There might some name and resolution conflicts between these two.
        @param datapath: original scannet folder path, default in yaml is '/media/zhongad/2TB/dataset/scannet'
        @param mode: train/val/test
        @param n_scales: number of coarse to fine level, used to read in the prepared data
        @param pklpath: Available as the meta path for different scene folders in scannet
        @param fragpath: Available when assigning a particular scene
        """

        # pklpath is dumped afterwards in this repo
        super(ScanNetDataset, self).__init__()
        self.datapath = datapath
        self.pklpath = pklpath
        self.mode = mode
        self.datasplit = datasplit
        self.n_views = nviews
        self.transforms = transforms

        if fragpath is not None:  # only for a particular scene
            self.metas = build_list_onset(fragpath)
        else:
            # a list of dict [{scene 0}, ..., {scene N}]
            self.metas = self.build_list()

        # give the correct source path to access the scannet files, rather than npz we processed
        if mode == 'test':
            self.source_path = 'scans_test'
        else:
            self.source_path = 'scans'

        self.n_scales = n_scales
        self.epoch = None
        self.tsdf_cache = {}
        self.semseg_cache = {}
        self.max_cache = 100

        # 2*src_on_either frames will not be considered as ref frame in this fragment
        self.n_src_on_either_side = n_src_on_either_side  # same as 3dvnet, for the whole fragment then, there should be 3 imgs to be the ref imgs
        self.crop = crop
        self.scale_rgb = scale_rgb
        self.mean_rgb = mean_rgb
        self.std_rgb = std_rgb

        self.depth_prediction = depth_prediction
        self.semseg_matching = semseg_matching
        self.link_augmentation = False
        self.nyu40_20_conversion = nyu40_20_conversion
        self.nyu40_to_20_label_mapper = -np.ones(41)
        for i, x in enumerate(VALID_CLASS_IDS_NYU40):
            self.nyu40_to_20_label_mapper[x] = i

        if self.depth_prediction:
            self.dbatch_path = '/media/zhongad/2TB/dataset/scannet/3dvnet_proc'
            self.dbatch_depth_size, self.dbatch_img_size, dbatch_augment = (56, 56),  (256, 320), False

    def build_list(self):
        # load the train/test/val fragments and assign scenes' fragments
        dataset = self.mode if self.datasplit is None else self.datasplit
        with open(os.path.join(self.pklpath, self.tsdf_file, 'fragments_{}.pkl'.format(dataset)), 'rb') as f:
            metas = pickle.load(f)

        if self.mode in ['val', 'test']:
            _save_mesh_resume = False
            if _save_mesh_resume:
                import glob
                logdir = '/home/zhongad/3D_workspace/SemanticCDR/results/scannet_logs_train_semseg-3phases-withfixing_fusion_eval_32'
                # metas_scene = [i.get('scene') for i in metas]
                current_scene = [i.split('/')[-1].split('.')[0] for i in sorted(glob.glob(logdir + '/*.txt'))]
                new_metas = []
                for i in range(len(metas)):
                    if metas[i]['scene'] not in current_scene:
                        new_metas.append(metas[i])
                print('after removing, len(metas) == {}'.format(len(new_metas)))
                return new_metas
        return metas

    def __len__(self):
        return len(self.metas)

    def read_scene_volumes(self, data_path, scene):
        """
        this fn is refactored to read in both tsdf and semseg. For each scene, only read tsdf into tsdf_cache for once.
        Thus for check scene in the tsdf_cache or not before read, for each iteration.
        @param data_path:
        @param scene:
        @return:
        """
        if scene not in self.tsdf_cache.keys():
            if len(self.tsdf_cache) > self.max_cache:
                self.tsdf_cache = {}
                self.semseg_cache = {}
            full_tsdf_list = []
            full_semseg_list = []
            for l in range(self.n_scales + 1):  # range(len(MODEL.THRESHOLDS) == 3)
                # load full tsdf volume
                full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(l)),
                                    allow_pickle=True)
                full_tsdf_list.append(full_tsdf.f.arr_0)
                full_semseg = np.load(os.path.join(data_path, scene, 'full_semseg_layer{}.npz'.format(l)),
                                      allow_pickle=True)
                full_semseg = full_semseg.f.arr_0
                if self.nyu40_20_conversion:
                    # map 41 labels into 20 valid labels in ScanNet official eval (0-19) only
                    full_semseg = self.nyu40_to_20_label_mapper[full_semseg]
                full_semseg_list.append(full_semseg)
            self.tsdf_cache[scene] = full_tsdf_list
            self.semseg_cache[scene] = full_semseg_list

        else:
            # print('no need to np load tsdf, already in cache')
            pass
        return self.tsdf_cache[scene], self.semseg_cache[scene]

    def read_frag_create_ref_src_frames(self, frag_scene_id, frag_img_ids):
        """
        This func creates 'dbatch' data object for each fragment loaded.
        Notice that at this point we read info from below in /3dvnet_proc to create 'dbatch' for reproductivity
        - color
        - depth
        - info.json

        @param frag_scene_id: fragment scene id, e.g., scenexxxx_xx
        @param frag_img_ids: images ids determined in the fragment.
        Cannot be directly used, must be changed into frag_img_info_ids, indices wrt dbatch_scene_info for further usage

        NOTE that these two params haven't been used yet, coz I need to use dbatch data to first get the correct depth_pred, before training a complete cdrnet.
        @return dbatch: the depth batch data object
        """
        dbatch_path = self.dbatch_path
        dbatch_scene_dir = os.path.join(dbatch_path, 'scans', frag_scene_id) \
            if self.mode in ['train', 'val'] else os.path.join(dbatch_path, 'scans_test', frag_scene_id)
        dbatch_scene_info = json.load(open(os.path.join(dbatch_scene_dir, 'info.json'), 'r'))
        info_img_ids_list = [int(d['filename_color'].split('/')[-1].split('.')[0]) for d in dbatch_scene_info['frames']]

        # change indices wrt dbatch_scene_info, but only a small portion of dataset has invalid pose `inf`
        #   thus doing this fix only a couple scenes, won't have too much influence on the training, but still good tho
        frag_img_ids = [info_img_ids_list.index(i) for i in frag_img_ids]

        dbatch_all_poses = np.stack([np.asarray(frame['pose']) for frame in dbatch_scene_info['frames']], axis=0)
        dbatch_K = np.asarray(dbatch_scene_info['intrinsics'])

        ref_idx = np.array(frag_img_ids[self.n_src_on_either_side:-self.n_src_on_either_side])
        n_ref_imgs = ref_idx.shape[0]
        n_imgs_per_ref = 2 * self.n_src_on_either_side + 1

        # used as a lut for ref-srcs pair in mvsnet, is it also for graph nn edge conv?
        # ref img and src img could be the same one
        ref_src_edges = torch.empty((2, n_ref_imgs * n_imgs_per_ref), dtype=torch.long)

        # all the ref/src edge img id pairs
        for i in range(n_ref_imgs):
            ref_src_edges[0, i * n_imgs_per_ref: (i + 1) * n_imgs_per_ref] = torch.ones(n_imgs_per_ref) * (
                    i + self.n_src_on_either_side)
            ref_src_edges[1, i * n_imgs_per_ref: (i + 1) * n_imgs_per_ref] = torch.arange(i, i + n_imgs_per_ref)

        # dbatch img depth preprocessing
        raw_images = []
        raw_depths = []
        for i in frag_img_ids:
            frame_info = dbatch_scene_info['frames'][i]
            color = cv2.imread(frame_info['filename_color'])
            depth = cv2.imread(frame_info['filename_depth'], cv2.IMREAD_ANYDEPTH)
            raw_images.append(color)
            raw_depths.append(depth)

        preprocessor_dbatch = PreprocessImage(K=dbatch_K,
                                              old_width=raw_images[0].shape[1],
                                              old_height=raw_images[0].shape[0],
                                              new_width=self.dbatch_img_size[1],
                                              new_height=self.dbatch_img_size[0],
                                              distortion_crop=0,
                                              perform_crop=self.crop)
        rgb_sum = 0
        intermediate_depths = []
        intermediate_images = []
        for i in range(len(raw_images)):
            depth = (raw_depths[i]).astype(np.float32) / 1000.0
            depth_nan = depth == np.nan
            depth_inf = depth == np.inf
            depth_outofrange = depth > 65.  # from 7scenes threshold
            depth_invalid = depth_inf | depth_nan | depth_outofrange
            depth[depth_invalid] = 0
            depth = preprocessor_dbatch.apply_depth(depth)
            intermediate_depths.append(depth)

            image = raw_images[i]
            image = preprocessor_dbatch.apply_rgb(image=image,
                                                  scale_rgb=1.0,
                                                  mean_rgb=[0.0, 0.0, 0.0],
                                                  std_rgb=[1.0, 1.0, 1.0],
                                                  normalize_colors=False)
            rgb_sum += np.sum(image)
            intermediate_images.append(image)
        rgb_average = rgb_sum / (len(raw_images) * self.dbatch_img_size[0] * self.dbatch_img_size[1] * 3)

        # # color augmentation
        # color_transforms = []
        # brightness = random.uniform(-0.03, 0.03)
        # contrast = random.uniform(0.8, 1.2)
        # gamma = random.uniform(0.8, 1.2)
        # color_transforms.append((adjust_gamma, gamma))
        # color_transforms.append((adjust_contrast, contrast))
        # color_transforms.append((adjust_brightness, brightness))
        # random.shuffle(color_transforms)

        K = preprocessor_dbatch.get_updated_intrinsics()
        depth_images = []
        images = []
        for i in range(len(raw_images)):
            image = intermediate_images[i]
            depth = intermediate_depths[i]
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image.astype(np.float32))
            image /= 255.0

            # if self.dbatch_augment and (55.0 < rgb_average < 200.0):
            #     for (color_transforms_func, color_transforms_val) in color_transforms:
            #         image = color_transforms_func(image, color_transforms_val)

            image = (image * 255.0) / self.scale_rgb
            image[0, ...] = (image[0, ...] - self.mean_rgb[0]) / self.std_rgb[0]
            image[1, ...] = (image[1, ...] - self.mean_rgb[1]) / self.std_rgb[1]
            image[2, ...] = (image[2, ...] - self.mean_rgb[1]) / self.std_rgb[2]

            images.append(image)

            depth = torch.from_numpy(depth.astype(np.float32))
            depth_images.append(depth)

        depth_images = torch.stack(depth_images, dim=0)
        images = torch.stack(images, dim=0)
        rotmats = torch.from_numpy(dbatch_all_poses[frag_img_ids, :3, :3]).float().transpose(2, 1)
        tvecs = -torch.bmm(rotmats, torch.from_numpy(dbatch_all_poses[frag_img_ids, :3, 3, None]).float())[..., 0]
        K = torch.from_numpy(K.astype(np.float32)).unsqueeze(0).expand(images.shape[0], 3, 3)
        if self.n_src_on_either_side > 0:
            depth_images = depth_images[self.n_src_on_either_side:-self.n_src_on_either_side]
        if (raw_depths[0].shape[0] != self.dbatch_depth_size[0]) or (raw_depths[0].shape[1] != self.dbatch_depth_size[1]):
            depth_images = F.interpolate(depth_images.unsqueeze(1), self.dbatch_depth_size, mode='nearest').squeeze(1)

        # # color augmentation, random rot about gravitational axis
        # R_aug = torch.from_numpy(random_gravitational_rotation_scannet()) if self.dbatch_augment \
        #     else torch.eye(3, dtype=torch.float32)
        # rotmats = rotmats @ R_aug.T
        #
        # # scale aug
        # S_aug = random.uniform(0.9, 1.1) if self.dbatch_augment else 1.
        # depth_images = depth_images * S_aug
        # tvecs = tvecs * S_aug

        dbatch = Batch(images, rotmats, tvecs, K, depth_images, ref_src_edges, raw_images)
        return dbatch, ref_idx.tolist()

    def __getitem__(self, idx):
        # fetch a fragment sample dict
        meta = self.metas[idx]

        imgs = []
        depths = []
        labels = []
        poses_list = []  # camera pose in scannet, in camera-to-world coord
        intrinsics_list = []

        tsdf_list, semseg_list = self.read_scene_volumes(self.pklpath, meta['scene'])

        for i, vid in enumerate(meta['image_ids']):
            # load imported
            imgs.append(
                read_img(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'color', '{}.jpg'.format(vid))))

            depths.append(
                read_depth(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'depth', '{}.png'.format(vid)))
            )
            labels.append(
                read_label(os.path.join(self.datapath, self.source_path, meta['scene'], 'label-filt-mapped',
                                        '{}.png'.format(vid)))
            )

            # load intrinsics and extrinsics
            intrinsics, poses = read_cam_file(os.path.join(self.datapath, self.source_path, meta['scene']),
                                              vid)
            intrinsics_list.append(intrinsics)
            poses_list.append(poses)

        intrinsics = np.stack(intrinsics_list)
        poses = np.stack(poses_list)

        items = {
            'imgs': imgs,
            'depths': depths,
            'labels': labels,  # raw id semantic label, note that to be consistent with bpnet, I used label-filt-mapped
            'intrinsics': intrinsics,
            'poses': poses,
            'tsdf_list_full': tsdf_list,  # [load(f/tl0.npz), load(1.npz), load(2.npz)] at read_scene_volumes()
            'semseg_list_full': semseg_list,  # 3d semseg with the same shape as TSDF
            'vol_origin': meta['vol_origin'],
            'scene': meta['scene'],
            'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
            'epoch': [self.epoch],
        }

        # read in semseg link under data/scannet/matching_mat
        if self.semseg_matching:
            matching_mat_name = meta['scene'] + '_frag' + str(meta['fragment_id']) + '.npz'
            matching_dict = np.load(os.path.join('./data/matching_mat', matching_mat_name),  allow_pickle=True)
            links, pth_voxelize_coords = matching_dict['links'], matching_dict['pth_voxelize_coords']
            items.update({
                'links_list': links.tolist(),  # using npz to store and load list automatically change into ndarray
                'pth_voxelize_coords_list': pth_voxelize_coords.tolist(),
            })

        if self.depth_prediction:
            dbatch, ref_frame_ids = self.read_frag_create_ref_src_frames(frag_scene_id=meta['scene'],
                                                                         frag_img_ids=meta['image_ids'])
            dbatch.__setattr__('images_batch_idx', torch.zeros(dbatch.images.shape[0], dtype=torch.long))
            items.update({
                'dbatch': dbatch
            })

        if self.transforms is not None:
            items = self.transforms(items)
        return items
    

class PreprocessImage:
    """ To process the raw scannet image, into 320x256 depth frame needed for 2d depth pred"""

    def __init__(self, K, old_width, old_height, new_width, new_height, distortion_crop=0, perform_crop=True):
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        self.new_width = new_width
        self.new_height = new_height
        self.perform_crop = perform_crop

        original_height = np.copy(old_height)
        original_width = np.copy(old_width)

        if self.perform_crop:
            old_height -= 2 * distortion_crop
            old_width -= 2 * distortion_crop

            old_aspect_ratio = float(old_width) / float(old_height)
            new_aspect_ratio = float(new_width) / float(new_height)

            if old_aspect_ratio > new_aspect_ratio:
                # we should crop horizontally to decrease image width
                target_width = old_height * new_aspect_ratio
                self.crop_x = int(np.floor((old_width - target_width) / 2.0)) + distortion_crop
                self.crop_y = distortion_crop
            else:
                # we should crop vertically to decrease image height
                target_height = old_width / new_aspect_ratio
                self.crop_x = distortion_crop
                self.crop_y = int(np.floor((old_height - target_height) / 2.0)) + distortion_crop

            self.cx -= self.crop_x
            self.cy -= self.crop_y
            intermediate_height = original_height - 2 * self.crop_y
            intermediate_width = original_width - 2 * self.crop_x

            factor_x = float(new_width) / float(intermediate_width)
            factor_y = float(new_height) / float(intermediate_height)

            self.fx *= factor_x
            self.fy *= factor_y
            self.cx *= factor_x
            self.cy *= factor_y
        else:
            self.crop_x = 0
            self.crop_y = 0
            factor_x = float(new_width) / float(original_width)
            factor_y = float(new_height) / float(original_height)

            self.fx *= factor_x
            self.fy *= factor_y
            self.cx *= factor_x
            self.cy *= factor_y

    def apply_depth(self, depth):
        raw_height, raw_width = depth.shape
        cropped_depth = depth[self.crop_y:raw_height - self.crop_y, self.crop_x:raw_width - self.crop_x]
        resized_cropped_depth = cv2.resize(cropped_depth, (self.new_width, self.new_height),
                                           interpolation=cv2.INTER_NEAREST)
        return resized_cropped_depth

    def apply_rgb(self, image, scale_rgb, mean_rgb, std_rgb, normalize_colors=True):
        raw_height, raw_width, _ = image.shape
        cropped_image = image[self.crop_y:raw_height - self.crop_y, self.crop_x:raw_width - self.crop_x, :]
        cropped_image = cv2.resize(cropped_image, (self.new_width, self.new_height), interpolation=cv2.INTER_LINEAR)

        if normalize_colors:
            cropped_image = cropped_image / scale_rgb
            cropped_image[:, :, 0] = (cropped_image[:, :, 0] - mean_rgb[0]) / std_rgb[0]
            cropped_image[:, :, 1] = (cropped_image[:, :, 1] - mean_rgb[1]) / std_rgb[1]
            cropped_image[:, :, 2] = (cropped_image[:, :, 2] - mean_rgb[2]) / std_rgb[2]
        return cropped_image

    def get_updated_intrinsics(self):
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]])


if __name__ == '__main__':
    # import wandb
    PKL_PATH = '../../Neucon-bare/data/scannet'
    cfg.defrost()
    cfg.merge_from_file('../configs/semseg_train_p1.yaml')
    cfg.freeze()

    # no augmentation for now, since read in raw matching matrix
    n_views = 9
    random_rotation = False
    random_translation = False
    paddingXY = 0
    paddingZ = 0

    transform = []
    transform += [transforms.ResizeImageAndLabel((640, 480)),
                  transforms.ToTensor(),
                  transforms.RandomTransformSpace(cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE, random_rotation,
                                                         random_translation, paddingXY, paddingZ),
                  transforms.IntrinsicsPoseToProjection(n_views, 4),
                  ]
    transforms = transforms.Compose(transform)
    transform_label_to_color = transforms.VizSemseg()
    train_dataset_tester = ScanNetDataset(cfg.TRAIN.PATH, "train", transforms,
                                          nviews=cfg.TRAIN.N_VIEWS,
                                          n_scales=len(cfg.MODEL.THRESHOLDS) - 1,
                                          pklpath=PKL_PATH)
