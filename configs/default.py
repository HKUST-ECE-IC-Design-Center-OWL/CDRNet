from yacs.config import CfgNode as CN

_C = CN()

_C.MODE = 'train'
_C.DATASET = 'scannet'
_C.PKL_PATH = 'data/meta_data'
_C.BATCH_SIZE = 1
_C.LOADCKPT = None
_C.LOGDIR = ''
_C.DEBUG = True
_C.WANDB = False
_C.WANDB_PROJ = ''
_C.WANDB_RUN_NAME = ''
_C.RESUME = True  # take the latest ckpt
_C.SUMMARY_FREQ = 20
_C.SAVE_FREQ = 1
_C.SEED = 42
_C.SAVE_SCENE_MESH = False
_C.SAVE_INCREMENTAL = False
_C.VIS_INCREMENTAL = False
_C.VIS_MESH_SEMSEG = True
_C.REDUCE_GPU_MEM = False
_C.RTMP_SERVER = None
_C.POSE_SERVER = None
_C.LOCAL_RANK = 0
_C.DISTRIBUTED = False

# test
_C.TEST = CN()
_C.TEST.PATH = ''
_C.TEST.N_VIEWS = 5
_C.TEST.N_WORKERS = 4
_C.TEST.DATASET_SPLIT = None

# model
_C.MODEL = CN()
_C.MODEL.N_VOX = [128, 224, 192]
_C.MODEL.VOXEL_SIZE = 0.04
_C.MODEL.THRESHOLDS = [0, 0, 0]
_C.MODEL.N_STAGE = 3
_C.MODEL.STAGE_LIST = ['coarse', 'medium', 'fine']
_C.MODEL.TRAIN_NUM_SAMPLE = [4096, 16384, 65536]
_C.MODEL.TEST_NUM_SAMPLE = [32768, 131072]
_C.MODEL.LW = [1.0, 0.8, 0.64]
_C.MODEL.PIXEL_MEAN = [103.53, 116.28, 123.675]
_C.MODEL.PIXEL_STD = [1., 1., 1.]
_C.MODEL.THRESHOLDS = [0, 0, 0]
_C.MODEL.POS_WEIGHT = 1.0
_C.MODEL.VIS_DEPTH = False
_C.MODEL.VIS_DEBUG_REFMNT = False
_C.MODEL.DEPTH_PREDICTION = False

_C.MODEL.BACKBONE2D = CN()
_C.MODEL.BACKBONE2D.ARC = 'fpn-mnas'
_C.MODEL.BACKBONE2D.CHANNELS = [96, 48, 24]  # c/m/f: 16th/8th/4th
_C.MODEL.SEMSEG_MULTISCALE = True
_C.MODEL.LW_SEMSEG = [1.0, 0.8, 0.64]  # semseg also needs to be multi-scale

_C.MODEL.SPARSEREG = CN()
_C.MODEL.SPARSEREG.DROPOUT = False

_C.MODEL.FUSION = CN()
_C.MODEL.FUSION.FUSION_ON = False  # control whether gru_fusion() is utilized in the coarse to fine network
_C.MODEL.FUSION.HIDDEN_DIM = 64
_C.MODEL.FUSION.AVERAGE = False
_C.MODEL.FUSION.FULL = False  # control whether to merge the local TSDF volume with the global one, if True, needs to update the grid_mask to calculate correct loss for the local, then valid_volume, and updated_coords
_C.MODEL.CDR = CN()
_C.MODEL.CDR.SEMSEG_LOSS_INCLUDE = True  # optimize on semseg loss
_C.MODEL.CDR.DEPTH_PRED = False  # mvsnet init depth prediction and pointflow depth refinement
_C.MODEL.CDR.FEAT_REFMNT = False  # feature refinement
_C.MODEL.CDR.SEMSEG_REFMNT = False  # semseg 2d link to 3d
_C.MODEL.CDR.SEMSEG_2D = False
_C.MODEL.CDR.SEMSEG_CLASS_3D = 41  # semseg 2d link to 3d
_C.MODEL.CDR.SEMSEG_CLASS_2D = 20  # to solve label unmatched error? RuntimeError: CUDA error: device-side assert triggered
_C.MODEL.CDR.N_ITERS = 1
_C.MODEL.CDR.OFFSETS = [0.05, 0.05, 0.025]
_C.MODEL.CDR.FEAT_DIM = 32

# 2D depth prediction options
_C.MODEL.CDR.IMG_SIZE = (480, 640)  # this is 256x320 from 3dvnet, not used for now. The dbatch seems ok with 480x640
# model dimension settings
_C.MODEL.CDR.CHANNEL_FEAT_DIM = 24  # this is the channel dim of feats_quarter, needed to be matched with the channel setup in FPN of backbone2d
_C.MODEL.CDR.GRID_EDGE_LEN = 0.08  # voxel resolution for scene-modeling step

# mod for dbatch in mvs setups
_C.MODEL.DEPTH_MVS = CN()  # some config for init depth pred with mvsnet
_C.MODEL.DEPTH_MVS.IMAGE_GT_SIZE = (480, 640)
_C.MODEL.DEPTH_MVS.DEPTH_GT_SIZE = (480, 640)
_C.MODEL.DEPTH_MVS.DEPTH_START = 0.5
_C.MODEL.DEPTH_MVS.DEPTH_INTERVAL = 0.05  # for the init depth pred from mvsnet, each interval is .05m
_C.MODEL.DEPTH_MVS.N_INTERVALS = 96 # 'depth_pred_size': (60, 80),  # resolution of feat_8, so that not oom
_C.MODEL.DEPTH_MVS.DEPTH_PRED_SIZE = (56, 56)  # resolution tested by 3dvnet, which shows that is optimal than feat_8


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
