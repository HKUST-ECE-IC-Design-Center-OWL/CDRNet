""" Author: @HaFred
This pkg serves for the arkit video demo, where the inputs are folders (/color, /intrinsics, /pose_from_quat). It is for each frames segment, because we need the real-time inference result from neucon to be extracted and saved. Thus, no csv or json file as a whole data index can be used in this case, it has to be folders where you can easily add on data.

* I put the key snippet closely effective as those in demo.py, with a comment format of:
    `xx as in demo`
"""
import argparse
import os
import sys
sys.path.append('/home/zhongad/3D_workspace/BestSemanticCDR/')

from cdrnet_real_time_demo.arkit_live_demo.data_reader import DataReader
from configs_realtime import update_config
from configs_realtime import cfg

if __name__ == '__main__':
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    parser = argparse.ArgumentParser(description='arkit demo on cdrnet')
    parser.add_argument('--cfg',
                        help='the config file name',
                        default='configs_realtime/arkit_video_demo.yaml',
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(cfg, args)  # update the default config according to the config file

    Reader = DataReader(cfg.RTMP_SERVER, cfg.POSE_SERVER, cfg)
    Reader.start_receive()
