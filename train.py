import os, sys

import numpy as np
import imageio
import skimage
import scipy

try:
    imread = imageio.v2.imread
except Exception:
    imread = imageio.imread

import torch

# Root directory of the project
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DATA_DIR = ROOT_DIR
if not ROOT_DIR in sys.path:
    sys.path.append(ROOT_DIR)  # To find local version of the library
    
DEFAULT_NO_OF_EPOCHS = 25

from mrcnn_torch.mrcnn_model import build_model, train
from rpe_config import RPE_Config
from rpe_dataset import RPE_Dataset, RPE_Augmenter

def parse_resp_file(resp_path):
    global DEFAULT_DATA_DIR
    fpath = os.path.abspath(resp_path)
    with open(fpath, 'rt') as fi:
        lines = fi.read().split('\n')
    arglist = []
    for _line in lines:
        line = _line.strip()
        if len(line) == 0 or line.startswith('#'):
            continue
        idx = line.find('=')
        if idx < 0:
            idx = line.find(' ')
        if idx < 0:
            arglist.append(line)
            continue
        arglist.append(line[:idx])
        arglist.append(line[idx+1:])
    DEFAULT_DATA_DIR = os.path.dirname(fpath)
    return arglist

if __name__ == '__main__':

    if len(sys.argv) > 1 and sys.argv[1].startswith('@'):
        try:
            arglist = parse_resp_file(sys.argv[1][1:])
        except Exception as ex:
            print ('Error parsing response file:', str(ex))
            sys.exit(1)
    else:
        arglist = sys.argv[1:]
    
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Pytorch-based Mask_RCNN model to segment RPE Map image stacks.')
    parser.add_argument('channel', nargs=1,
            metavar='CHANNEL',
            help='Microscope Channel Label (DNA, Actin, Actin-BW, etc.)')
    parser.add_argument('-d', '--data-dir', required=False,
            metavar="/data/directory",
            default=DEFAULT_DATA_DIR,
            help='Directory where to look for data: model_weights, training_data, etc.\nDefault: "%s"' % \
                DEFAULT_DATA_DIR)
    parser.add_argument('-t', '--training-data', required=False,
            metavar='/path/to/train/data',
            default=RPE_Config.TRAINING_SUBDIR,
            help='Train Data Directory, can be relative to {data-dir}. Must contain {CHANNEL}/imgs and {CHANNEL}/masks.\n'+\
            'Default: "%s/".' % (RPE_Config.TRAINING_SUBDIR,))
    parser.add_argument('-w', '--weights', required=False,
            metavar="/path/to/weights",
            default=RPE_Config.WEIGHTS_SUBDIR,
            help='Path to weights .pth file or directory containing .pth.\n'+\
                'Can be relative to {data-dir}. Default: "%s/".' % (RPE_Config.WEIGHTS_SUBDIR,))
    parser.add_argument('-l', '--logdir', required=False,
            metavar="/path/to/logs",
            default='logs',
            help='Path to directory where to write logs.\n'+\
                'Can be relative to {data-dir}. Default: "logs/".')
    parser.add_argument('-e', '--epochs', required=False, type=int,
            metavar="NO_OF_EPOCHS",
            default=DEFAULT_NO_OF_EPOCHS,
            help="Number of training epochs (default: %d)" % DEFAULT_NO_OF_EPOCHS)
    parser.add_argument('-r', '--rate', required=False, type=float,
            metavar="LEANING_RATE",
            default=RPE_Config.learning_rate,
            help="Learning rate (default: %f)" % RPE_Config.learning_rate)
    parser.add_argument('-m', '--momentum', required=False, type=float,
            metavar="MOMENTUM",
            default=RPE_Config.momentum,
            help="Learning momentum (default: %f)" % RPE_Config.momentum)
    parser.add_argument('-D', '--disable-gpu', required=False, action="store_true",
            help="Disable GPU(s), e.g. if GPUs have insufficient memory and the script crashes.")

    args = parser.parse_args(arglist)
    args.data_dir = os.path.abspath(os.path.join(DEFAULT_DATA_DIR, args.data_dir))
    args.training_data = os.path.abspath(os.path.join(args.data_dir, args.training_data))
    args.weights = os.path.abspath(os.path.join(args.data_dir, args.weights))
    args.logdir = os.path.abspath(os.path.join(args.data_dir, args.logdir))
    
    if args.disable_gpu:
        print ('Disabling GPUs (if any).')
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        
    print(args)

    channel = args.channel[0]
    img_type = 'RGB' if channel=='Actin' else 'BW'
    parts = channel.split('-')
    if len(parts) > 1:
        channel = parts[0]
        img_type = parts[1].upper()
        assert img_type in ('BW', 'RGB'), 'Image type must be BW or RGB'

    cfg = RPE_Config(args.training_data, channel, img_type)
    
    os.makedirs(args.weights, exist_ok=True)
    assert os.path.isdir(args.weights), f"Can't access model weights directory {args.weights}"
    
    cfg.learning_rate = args.rate
    cfg.momentum = args.momentum
    cfg.gamma = 0.316227766
    cfg.step_size = 25
    cfg.min_ann_per_img = 1
    cfg.trainable_layers = 5

    tdir = os.path.join(args.training_data, 'Mask_RCNN', f'{cfg.class_name}-{cfg.image_type}')
    if not os.path.isdir(tdir) and cfg.is_default_image_type:
        # Directory with explicit image type not found, try default if applicable
        tdir = os.path.join(args.training_data, 'Mask_RCNN', cfg.class_name)
    assert os.path.isdir(tdir), f"Can't access training data directory {tdir}"
    ndds = RPE_Dataset(tdir, skip=None)
    assert len(ndds) >= 10, 'Not enough training data in '+tdir+' - need at least 10 items.'
    
    augm = RPE_Augmenter()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device {device}')

    model = build_model(
        num_classes=cfg.num_classes,
        detections_per_img=cfg.detections_per_img,
        score_thresh=cfg.score_thresh,
        trainable_layers=cfg.trainable_layers
        )
    
    model.to(device)
    model.train()
    train(model, device, cfg, ndds, args.weights, augm=augm.augment, num_epochs=args.epochs, logdir=args.logdir)

    print('Done training.')
    sys.exit(0)