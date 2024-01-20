import argparse
import os
import time
import datetime
import torch
import trimesh
import numpy as np
import torch.nn.functional as F
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader  # for depth Batch
from loguru import logger
from utils import tensor2float, make_nograd_func, SaveScene
from datasets import transforms
from configs import cfg, update_config
from eval.evaluation import Renderer
from datasets.scannet import ScanNetDataset
from src.cdrnet import CrossDimensionalRefmntNet


def args():
    parser = argparse.ArgumentParser(description='Testing for CDRNet model')
    parser.add_argument('--cfg',
                        help='experiment configuration file',
                        default='configs/inference_scene0249.yaml',
                        type=str)
    parser.add_argument('--save_path', type=str, default=None)
    arguments = parser.parse_args()
    return arguments


args_object = args()
update_config(cfg, args_object)

# for determinism
torch.manual_seed(cfg.SEED)
torch.cuda.manual_seed(cfg.SEED)

# create logger
current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
if not cfg.DEBUG:
    logdir = cfg.LOGDIR
    if not os.path.isdir(cfg.LOGDIR):
        os.makedirs(cfg.LOGDIR)
    elif cfg.LOGDIR is None:
        os.makedirs(f'inference_{current_time_str}', exist_ok=False)
        logdir = f'inference_{current_time_str}'
    logfile_path = os.path.join(logdir, f'{current_time_str}_{cfg.MODE}.log')
    logger.add(logfile_path, format='{time} {level} {message}', level="INFO", rotation="12:00")

# data augmentation
n_views = cfg.TEST.N_VIEWS
random_rotation = False
random_translation = False
paddingXY = 0
paddingZ = 0
transform = []
transform += [transforms.ResizeImageAndLabel((640, 480)),
              transforms.ToTensor(),
              transforms.RandomTransformSpace(
                  cfg.MODEL.N_VOX,
                  cfg.MODEL.VOXEL_SIZE,
                  random_rotation,
                  random_translation,
                  paddingXY,
                  paddingZ),
              transforms.IntrinsicsPoseToProjection(n_views, 4)]
transforms = transforms.Compose(transform)

# single scene dataset
single_scene_scannet = ScanNetDataset(datapath=cfg.TEST.PATH,
                                      fragpath='data/meta_data/scene0249_00/fragments.pkl',
                                      pklpath=cfg.PKL_PATH,
                                      mode='val',
                                      transforms=transforms,
                                      nviews=cfg.TEST.N_VIEWS,
                                      n_scales=len(cfg.MODEL.THRESHOLDS) - 1,
                                      datasplit='val',
                                      depth_prediction=cfg.MODEL.DEPTH_PREDICTION,
                                      semseg_matching=cfg.MODEL.CDR.SEMSEG_REFMNT,
                                      nyu40_20_conversion=cfg.MODEL.CDR.SEMSEG_CLASS_3D == 20
                                      )
inference_loader = DataLoader(single_scene_scannet,
                              batch_size=cfg.BATCH_SIZE,
                              num_workers=cfg.TEST.N_WORKERS,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=False)

model = CrossDimensionalRefmntNet(cfg).cuda().eval()
model = torch.nn.DataParallel(model, device_ids=[0])
model_49 = CrossDimensionalRefmntNet(cfg).cuda().eval()
model_49 = torch.nn.DataParallel(model_49, device_ids=[0])
save_mesh = SaveScene(cfg)
logger.info(model)
logger.info('Cfg is as below: {}\n'.format(cfg))


@make_nograd_func
def inference_sample(sample, save_scene=False, duration=None):
    model.eval()
    start_time = time.time()
    outputs, vis_metrics = model(sample, save_scene)
    duration += time.time() - start_time
    loss = vis_metrics['total_loss']
    tsdf_loss = vis_metrics['total_tsdf_occ_loss']
    if 'total_3d_semseg_loss' in vis_metrics.keys():
        semseg_loss = vis_metrics['total_3d_semseg_loss']

    if 'total_3d_semseg_loss' in vis_metrics.keys():
        return tensor2float(loss), outputs, tensor2float(tsdf_loss), tensor2float(semseg_loss), duration


def inference(loadckpt):
    logger.info('resuming ' + str(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'], strict=False)
    # state_dict_49 = torch.load('model_000049.ckpt')
    # model_49.load_state_dict(state_dict_49, strict=False)
    epoch_idx = state_dict['epoch']
    logger.info('Conducting inference, with model epoch {}'.format(epoch_idx))
    batch_len = len(inference_loader)
    tsdf_loss_list = []
    semseg_loss_list = []
    total_loss_list = []
    duration_time = 0

    if cfg.MODEL.VIS_DEPTH:  # to create renderer based on the predicted mesh, vis depth needed to be one secondly
        mesh_to_render = 'logs_viz_one_scene_valset/val/scene_scannet_fusion_eval_49_with_depth_jan17/scene0249_00.ply'
        mesh_to_render = trimesh.load(mesh_to_render, process=False)
        renderer = Renderer()
        mesh_opengl = renderer.mesh_opengl(mesh_to_render)

    with torch.no_grad():
        for batch_idx, sample in enumerate(inference_loader):
            save_scene = cfg.SAVE_SCENE_MESH and batch_idx == batch_len - 1
            if cfg.MODEL.VIS_DEPTH:  # to create the depth_from_mesh tensor
                h, w = 480, 640
                cam_intr = sample['intrinsics']
                cam_pose = sample['poses']
                depth_from_mesh_list = []
                for i in range(9):
                    _, depth_from_mesh = renderer(h, w,
                                                  cam_intr[0, i], cam_pose[0, i],
                                                  mesh_opengl)
                    depth_from_mesh_list.append(depth_from_mesh)
                depth_from_mesh = torch.FloatTensor(np.stack(depth_from_mesh_list))
                depth_from_mesh = F.interpolate(depth_from_mesh.unsqueeze(0), scale_factor=1 / 4,
                                                mode='nearest').squeeze(0)
                sample['depth_from_mesh'] = depth_from_mesh
            start_time = time.time()
            loss, outputs, tsdf_loss, semseg_loss, duration_time = \
                inference_sample(sample, save_scene=save_scene, duration=duration_time)
            logger.info('{}, epoch {}, iter {}/{}, tsdf loss = {:.3f}, semseg loss = {:.3f}, '
                        'time = {:.3f}'.format(sample['scene'][0], epoch_idx, batch_idx,
                                               len(inference_loader), tsdf_loss, semseg_loss,
                                               time.time() - start_time))
            if os.path.isfile(loadckpt):
                save_mesh(outputs, sample, epoch_idx)
            tsdf_loss_list.append(tsdf_loss)
            semseg_loss_list.append(semseg_loss)
            total_loss_list.append(loss)
            del tsdf_loss, semseg_loss, loss
        loss_mean_epoch = sum(total_loss_list) / len(total_loss_list)
        tsdf_mean_epoch = sum(tsdf_loss_list) / len(tsdf_loss_list)
        semseg_mean_epoch = sum(semseg_loss_list) / len(semseg_loss_list)
    return loss_mean_epoch, tsdf_mean_epoch, semseg_mean_epoch, duration_time


if __name__ == '__main__':
    frag_len = len(inference_loader)
    loss_mean, tsdf_mean, semseg_mean, duration = inference(loadckpt=cfg.LOADCKPT)
    summary_text = f"""
            Summary:
                Total number of fragments: {frag_len} 
                Total tsdf loss: {tsdf_mean}
                Total semseg loss: {semseg_mean}
                Total loss: {loss_mean}
                Average keyframes/sec: {1 / (duration / (frag_len * cfg.TEST.N_VIEWS))}
                Total runtime: {duration}
            """
    logger.info(summary_text)
