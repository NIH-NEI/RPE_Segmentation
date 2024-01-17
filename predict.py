import os, sys

import numpy as np
import tifffile
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

from mrcnn_torch.mrcnn_model import build_model, predict_one, predict_batch, train
from rpe_stack import RpeStack
from tiletools import slice_area
from savgol import SavitzkyGolay2D
from rpe_config import RPE_Config
from rpe_dataset import RPE_Dataset

import imagetools

def iter_stacks(root_dir, recurse=True):
    subdirs = []
    for fn in os.listdir(root_dir):
        fpath = os.path.join(root_dir, fn)
        if os.path.isdir(fpath):
            subdirs.append(fpath)
            continue
        fnlo = fn.lower()
        if fnlo.endswith('.rpe.json') or fnlo.endswith('.ome.tif'):
            yield fpath
    if recurse:
        for cdir in subdirs:
            for fpath in iter_stacks(cdir, recurse=False):
                yield fpath
#

def iter_multi_stacks(args, recurse=True):
    for spath in args.rpefile:
        fpath = os.path.abspath(os.path.join(args.data_dir, spath))
        if os.path.isfile(fpath):
            yield fpath
        elif os.path.isdir(fpath):
            root_dir = fpath
            for fpath in iter_stacks(root_dir, recurse):
                yield fpath
        else:
            print('Warning: no such file or directory:', fpath)
#

class RPE_Predictor(object):
    RGB_CHANNELS = ('DNA', 'Actin', 'Membrane')
    ALL_CHANNELS = ('DNA', 'Actin')
    AUTO_ADJUST = {
        'DNA':imagetools.POSTPROC_DNA,
        'Actin':imagetools.POSTPROC_ACTIN,
        }
    def __init__(self, args):
        self.args = args
        #
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.sg = SavitzkyGolay2D(5, 2)
        #
        self.cfg = cfg = RPE_Config(self.args.data_dir, 'Any', 'RGB')
        cfg.PREDICTIONS_SUBDIR = args.prediction_dir
        cfg.score_thresh = 0.65
        cfg.prob_thresh = 0.7
        cfg.good_iou=0.6
        cfg.ok_iou=0.5
        self.chlist = list(self.ALL_CHANNELS) if self.args.channel == 'All' else [self.args.channel,]
        #
        self.otsu3 = 16383.
        self.smooth_kern = self.sg.kernel(0, 0)
        self.norm_map = {}
        self.otsu = 16383.
        self.sc_norm = 1.
        #
        self.model_cache = {}
    #
    def predict_all(self):
        for fpath in iter_multi_stacks(self.args):
            fstk = RpeStack(fpath, default_channel=self.chlist[0])
            for chname in self.chlist:
                if not fstk.hasChannel(chname):
                    print(f'Warning: No channel {chname} in {fpath}.')
                    continue
                tgtdir = os.path.abspath(os.path.join(fstk.base_dir, self.cfg.PREDICTIONS_SUBDIR))
                #
                self.cfg.class_name = chname
                self.cfg.image_type = self.cfg.defaultImageType(self.cfg.class_name)
                if self.cfg.image_type == 'RGB' and not fstk.hasChannels(self.RGB_CHANNELS):
                    print('One or more RGB channels is missing, using BW model.')
                    self.cfg.image_type == 'BW'
                if os.path.isfile(self.args.weights):
                    wpath = os.path.abspath(self.args.weights)
                else:
                    epoch, wpath = self.cfg.find_model_weights(self.args.weights)
                    if wpath is None and self.cfg.image_type == 'RGB':
                        print('RGB model is missing, will try BW.')
                        self.cfg.image_type == 'BW'
                        epoch, wpath = cfg.find_model_weights(self.args.weights)
                if wpath is None:
                    print(f'Warning: no model weights for {chname}/{BW}')
                    continue
                os.makedirs(tgtdir, exist_ok=True)
                if not os.path.isdir(tgtdir):
                    print(f'Cannot access output directory {tgtdir}')
                    break
                #
                if self.args.adjust == 'Auto':
                    self.cfg.postproc = self.AUTO_ADJUST.get(self.cfg.class_name, imagetools.POSTPROC_NONE)
                else:
                    self.cfg.postproc = self.AUTO_ADJUST.get(self.args.adjust, imagetools.POSTPROC_NONE)
                print(f'Segmenting {chname} of {fpath}...')
                #
                self.predict_one(fstk, wpath, tgtdir)
    #
    def predict_one(self, fstk, wpath, tgtdir):
        chname = self.cfg.class_name
        model_key = (chname, self.cfg.image_type)
        if model_key in self.model_cache:
            model = self.model_cache[model_key]
        else:
            model = build_model(
                num_classes=self.cfg.num_classes,
                detections_per_img=self.cfg.detections_per_img,
                score_thresh=self.cfg.score_thresh)
            print('Loading model weights from:', wpath)
            model.load_state_dict(torch.load(wpath))
            model.to(self.device)
            model.eval()
            self.model_cache[model_key] = model
        #
        if self.cfg.postproc & imagetools.POSTPROC_DNA != 0:
            self.otsu3 = fstk.getChannelOtsu3(chname)[0]
        self.norm_map = {}
        if self.cfg.image_type == 'RGB':
            for _chname in self.RGB_CHANNELS:
                otsu = fstk.getChannelOtsu(_chname)
                sc_norm = 63./otsu
                self.norm_map[_chname] = sc_norm
        else:
            self.otsu = fstk.getChannelOtsu(chname)
            self.sc_norm = 16383./self.otsu
        #
        tiles = slice_area((fstk.height, fstk.width), (self.cfg.tilesize, self.cfg.tilesize))
        m_frames = np.zeros(shape=(fstk.n_frames, fstk.height, fstk.width), dtype=np.uint8)
        #
        particles_3d = []
        for cframe in range(fstk.n_frames):
            print(f'  frame {cframe+1} of {fstk.n_frames}')
            particles_2d = []
            #
            if self.cfg.image_type == 'RGB':
                fr_data, fr_mask = self.prepare_rgb_frame(fstk, cframe, chname)
            else:
                fr_data, fr_mask = self.prepare_bw_frame(fstk, cframe, chname)
            if not fr_mask is None:
                m_frames[cframe] = fr_mask
                del fr_mask
            #
            ndimgs = []
            origs = []
            for x0, x1, y0, y1 in tiles:
                ndimg = fr_data[y0:y1+1, x0:x1+1, :]
                masks, scores = predict_one(model, self.device, ndimg, prob_thresh=self.cfg.prob_thresh)
                if len(masks.shape) != 3:
                    nmasks = np.empty(shape=(1, self.cfg.tilesize, self.cfg.tilesize), dtype=np.uint8)
                    nmasks[0] = masks
                    masks = nmasks
                masks[masks!=0] = 0xFF
                particles = imagetools.masks_to_particles(masks, x0, y0)
                particles_2d.extend(particles)
            #
            particles_3d.append(particles_2d)
        #
        tbn = f'{fstk.base_name}_{chname}_RPE'
        csvname = os.path.join(tgtdir, tbn+'.csv')
        tifname = os.path.join(tgtdir, tbn+'.csv')
        print('3D assembly...')
        imagetools.assemble_ml(particles_3d, m_frames, csvname, self.cfg.postproc, self.cfg.good_iou, self.cfg.ok_iou)
        print('Write:', tifname)
        tifffile.imwrite(tifname, m_frames, photometric='minisblack')
    #
    def prepare_rgb_frame(self, fstk, cframe, chname):
        fr_data = np.empty(shape=(fstk.height, fstk.width, 3), dtype=np.uint8)
        fr_mask = None
        postproc = self.cfg.postproc & imagetools.POSTPROC_DNA != 0
        for i,_chname in enumerate(RPE_Predictor.RGB_CHANNELS):
            norm = self.norm_map[_chname]
            ch_data = fstk.getFrame(cframe, _chname).astype(np.float32)
            #
            if postproc and _chname == chname:
                fr_smooth = scipy.signal.convolve2d(ch_data, self.smooth_kern, boundary='symm', mode='same')
                fr_mask = np.zeros(shape=(fstk.height, fstk.width), dtype=np.uint8)
                fr_mask[fr_smooth > self.otsu3] = 0xFF
                del fr_smooth
            #   
            ch_data = ch_data * norm
            ch_data[ch_data > 255.] = 255.
            fr_data[:,:,i] = ch_data.astype(np.uint8)
        #
        return fr_data, fr_mask
    #
    def prepare_bw_frame(self, fstk, cframe, chname):
        fr_data = fstk.getFrame(cframe, chname).astype(np.float32)
        fr_mask = None
        if self.cfg.postproc & imagetools.POSTPROC_DNA != 0:
            fr_smooth = scipy.signal.convolve2d(fr_data, self.smooth_kern, boundary='symm', mode='same')
            fr_mask = np.zeros(shape=(fstk.height, fstk.width), dtype=np.uint8)
            fr_mask[fr_smooth > self.otsu3] = 0xFF
            del fr_smooth
        #
        fr_data = fr_data * self.sc_norm
        fr_data[fr_data > 65530.] = 65530.
        fr_data = skimage.color.gray2rgb(fr_data.astype(np.uint16))
        #
        return fr_data, fr_mask
    #


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
        description='Segment RPE Map image stacks using trained Pytorch-based Mask_RCNN model.')
    parser.add_argument('channel', nargs=1,
            metavar='CHANNEL',
            help='Microscope Channel Label (DNA, Actin or All)')
    parser.add_argument('rpefile', nargs='+',
            help='RPE Meta File .rpe.json, OME TIFF .ome.tif (or directory containing .rpe.json/.ome.tif)\n' +\
                'Can be relative to {data-dir}.')
    parser.add_argument('-p', '--prediction-dir', required=False,
            metavar="prediction/directory",
            default=RPE_Config.PREDICTIONS_SUBDIR,
            help='Output directory, relative to the directory or input files.\nDefault: "%s"' % \
                RPE_Config.PREDICTIONS_SUBDIR)
    parser.add_argument('-d', '--data-dir', required=False,
            metavar="/data/directory",
            default=DEFAULT_DATA_DIR,
            help='Default directory where to look for data: model_weights, rpefile, etc.\nDefault: "%s"' % \
                DEFAULT_DATA_DIR)
    parser.add_argument('-w', '--weights', required=False,
            metavar="/path/to/weights",
            default=RPE_Config.WEIGHTS_SUBDIR,
            help='Path to weights .pth file or directory containing .pth.\n'+\
                'Can be relative to {data-dir}. Default: "%s/".' % (RPE_Config.WEIGHTS_SUBDIR,))
    parser.add_argument('-t', '--tile-size', required=False, type=int,
            metavar="TILE_SIZE",
            default=RPE_Config.tilesize,
            help="Split large images into tiles of this size. If 0, use full frame images.\n" +\
            "Default: %d." % RPE_Config.tilesize)
    parser.add_argument('-a', '--adjust', required=False,
            metavar="DNA|Actin|Auto|None",
            default='Auto',
            help='Adjust particle borders (post processing). Possible values DNA, Actin, Auto or None.\n' +\
                '"Auto" is "DNA" for DNA Channel, "Actin" for Actin channel, "None" for anything else.')
    parser.add_argument('-D', '--disable-gpu', required=False, action="store_true",
            help="Disable GPU(s), e.g. if GPUs have insufficient memory and the script crashes.")

    args = parser.parse_args(arglist)
    args.data_dir = os.path.abspath(os.path.join(DEFAULT_DATA_DIR, args.data_dir))
    assert os.path.isdir(args.data_dir), '--data-dir must point to an existing directory.'
    if not args.adjust in ('DNA', 'Actin', 'Auto', 'None'):
        print('Warning: invalid value for "--adjust", ignored. Acceptable values are Auto, DNA, Actin or None.')
        args.adjust = 'None'
    args.weights = os.path.abspath(os.path.join(args.data_dir, args.weights))
    
    if args.disable_gpu:
        print ('Disabling GPUs (if any).')
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        
    args.channel = args.channel[0]
        
    print(args)
    
    pre = RPE_Predictor(args)
    pre.predict_all()

    print('Done, exiting(0).')
    sys.exit(0)