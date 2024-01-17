import os, sys, datetime
import json
import csv
import xml.etree.ElementTree as ET

import numpy as np
from tifffile import TiffFile, imwrite, imread
import skimage.filters as ski_filters

class RpeStackPage(object):
    def __init__(self, fpath, shape, dtype):
        self.fpath = fpath
        self.shape = shape
        self.dtype = dtype
        #
        self.tags = {'RpeStackPage':True}
    #
    def asarray(self):
        try:
            fr_data = imread(self.fpath)
            self.shape = fr_data.shape
            self.dtype = fr_data.dtype
            self.tags['valid'] = True
            return fr_data
        except Exception:
            pass
        self.tags['valid'] = False
        try:
            return np.zeros(self.shape, self.dtype)
        except Exception:
            return None
#

class RpeJsonStack(object):
    def __init__(self, fpath, fmeta=None):
        self.fpath = fpath
        #
        self.basedir = os.path.abspath(os.path.dirname(fpath))
        #
        self.n_frames = 0
        self.n_channels = 0
        self.dimorder = ''
        self.shape = None
        self.dtype = None
        self.pages = []
        filelist = []
        try:
            if fmeta is None:
                with open(self.fpath, 'r') as fi:
                    o = json.load(fi)
            else:
                o = fmeta
            self.n_frames = o['n_frames']
            self.n_channels = o['n_channels']
            self.dimorder = o['dimorder']
            for rpath in o['filelist']:
                if rpath:
                    fpath = os.path.abspath(os.path.join(self.basedir, rpath))
                else:
                    fpath = None
                filelist.append(fpath)
            self.labels = o['labels']
        except Exception:
            pass
        for fpath in filelist:
            try:
                fr_data = imread(fpath)
                self.shape = fr_data.shape
                self.dtype = fr_data.dtype
                break
            except Exception:
                pass
        else:
            self.nframes = self.nchannels = 0
            return
        self.pages = [RpeStackPage(fpath, self.shape, self.dtype) for fpath in filelist]
    #

#
# OME TIFF support:
#
# <xml>
# <Image ID="0" Name="IMAGE0">
#     <Pixels BigEndian="true" DimensionOrder="XYCZT" ID="0" SizeC="4" SizeT="1" SizeX="1278" SizeY="1078" SizeZ="27" Type="uint16">
#     </Pixels>
# </Image>
# </xml>
#
# Required: DimensionOrder, SizeC, SizeZ
#
class RpeStack(object):
    RGB_CHANNELS = ['DNA', 'Actin', 'Membrane']
    #
    def __init__(self, fpath, default_channel='DNA'):
        if isinstance(fpath, dict):
            self.fmeta = fmeta = fpath
            self.fpath = os.path.join(fmeta.pop('basedir'), fmeta.pop('basename')+'.rpe.json')
        else:
            self.fmeta = fmeta = None
            self.fpath = fpath
        self.default_channel = default_channel
        #
        self.fpath = os.path.abspath(self.fpath)
        self.base_dir, self.name = os.path.split(self.fpath)
        self.base_name, self.ext = os.path.splitext(self.name)
        if self.base_name.endswith('.ome') or self.base_name.endswith('.rpe'):
            self.base_name = self.base_name[:-4]
        #
        if self.ext.lower() == '.json':
            self.tif = RpeJsonStack(self.fpath, self.fmeta)
        else:
            self.tif = TiffFile(self.fpath)
        self.n_pages = len(self.tif.pages)
        self.n_frames = 0
        self.n_channels = 0
        self.dimorder = ''
        self.height = 0
        self.width = 0
        #
        self.shape = None
        self.dtype = None
        self.tags = {}
        if self.n_pages > 0:
            # self.nframes = self.n_pages / 4
            page = self.tif.pages[0]
            self.tags = page.tags
            self.shape = page.shape
            self.height = self.shape[0]
            self.width = self.shape[1]
            self.dtype = page.dtype
            self._parse_ome()
            self.dmo = self.dimorder.upper()[:4]
        #
        if self.n_frames > 0:
            if self.n_channels == 4:
                if hasattr(self.tif, 'labels') and len(self.tif.labels) == 4:
                    self.channels = self.tif.labels
                else:
                    self.channels = ['GFP', 'DNA', 'Actin', 'Membrane',]
            elif self.n_channels == 1:
                self.channels = [self.default_channel,]
        #
    #
    def _fridx(self, iframe, ichan):
        if self.dmo == 'XYZC':
            return ichan*self.n_frames + iframe
        return iframe*self.n_channels + ichan
    #
    def _parse_ome(self):
        try:
            tag = self.tags['RpeStackPage']
            self.dimorder = self.tif.dimorder
            self.n_channels = self.tif.n_channels
            self.n_frames = self.tif.n_frames
            return
        except Exception:
            pass
        try:
            tag = self.tags['ImageDescription']
            elem = et_find_elem(ET.fromstring(tag.value), ['Image', 'Pixels'])
            self.dimorder = elem.get('DimensionOrder')
            self.n_channels = int(elem.get('SizeC'))
            self.n_frames = int(elem.get('SizeZ'))
        except Exception:
            self.dimorder = ''
            self.n_channels = 1
            self.n_frames = self.n_pages
    #
    def saveMeta(self):
        if self.fmeta is None:
            return None
        try:
            with open(self.fpath, 'w') as fo:
                json.dump(self.fmeta, fo, indent=2)
            return self.fpath
        except Exception as ex:
            print (ex)
        return None
    #
    def channelName(self, ichan):
        try:
            chname = self._chlist[ichan]
        except Exception:
            chname = 'ch%02d' % (ichan,)
        return chname
    def channelId(self, chname, validated=False):
        if chname in self._chmap:
            return self._chmap[chname]
        if validated:
            raise ValueError('No such channel: '+chname)
        return -1
    def hasChannel(self, chname):
        return chname in self._chmap
    def hasChannels(self, chlist):
        for chname in chlist:
            if not self.hasChannel(chname):
                return False
        return True
    #
    @property
    def channels(self):
        return [self.channelName(ichan) for ichan in range(self.n_channels)]
    @channels.setter
    def channels(self, ch_list):
        self._chlist = [str(ch) for ch in ch_list]
        # self._chmap = dict([(self.channelName(ichan), ichan) for ichan in range(self.n_channels)])
        self._chmap = {}
        for ichan, ch in enumerate(self._chlist):
            self._chmap[ichan] = ichan
            self._chmap[ch] = ichan
            self._chmap[ch.replace('0', 'O')] = ichan
            self._chmap[ch.replace('O', '0')] = ichan
    #
    def getChannelData(self, chname=None):
        if chname is None:
            chname = self.default_channel
        ichan = self.channelId(chname, True)
        datayz = np.empty(shape=(self.n_frames, self.height, self.width), dtype=self.dtype)
        for iframe in range(self.n_frames):
            page = self.tif.pages[self._fridx(iframe, ichan)]
            datayz[iframe] = page.asarray()
        return datayz
    #
    def getChannelOtsu(self, chname=None):
        ichan = self.channelId(chname, True)
        chname = self.channelName(ichan)
        ch_data = np.float32(self.getChannelData(ichan))
        otsu = float(ski_filters.threshold_otsu(ch_data, 4096))
        return otsu
    #
    def getChannelOtsu3(self, chname=None):
        ichan = self.channelId(chname, True)
        chname = self.channelName(ichan)
        ch_data = np.float32(self.getChannelData(ichan))
        motsu = ski_filters.threshold_multiotsu(ch_data, classes=3, nbins=512)
        otsu3 = (float(motsu[0]), float(motsu[1]))
        return otsu3
    #
    def getFrame(self, iframe, chname=None):
        if chname is None:
            chname = self.default_channel
        ichan = self.channelId(chname, True)
        page = self.tif.pages[self._fridx(iframe, ichan)]
        data = page.asarray()
        return data
    #
    def subdir(self, name='Predicted'):
        return os.path.join(self.base_dir, name)
    def tname(self, chname, ext='.csv'):
        return f'{self.base_name}_{chname}_RPE{ext}'
    #
    def _rgb_norm_map(self):
        if hasattr(self, 'norm_map'):
            return self.norm_map
        self.norm_map = {}
        for _chname in self.RGB_CHANNELS:
            otsu = self.getChannelOtsu(_chname)
            self.norm_map[_chname] = 1. / otsu
        return self.norm_map
    def getRgbFrame(self, iframe):
        fr_data = np.empty(shape=(self.height, self.width, 3), dtype=np.uint8)
        norm_map = self._rgb_norm_map()
        for i,_chname in enumerate(self.RGB_CHANNELS):
            norm = 63. * norm_map[_chname]
            ch_data = self.getFrame(iframe, _chname).astype(np.float32)
            ch_data = ch_data * norm
            ch_data[ch_data > 255.] = 255.
            fr_data[:,:,i] = ch_data.astype(np.uint8)
        return fr_data
    #
    @staticmethod
    def is_acceptable_name(fn):
        base, ext = os.path.splitext(fn.lower())
        if ext in ('.tif', '.tiff'):
            return True
        if not ext in ('.json',):
            return False
        return base.endswith('.rpe')
    #
    @staticmethod
    def channelFromFilename(fpath):
        try:
            fn, ext = os.path.splitext(os.path.basename(fpath))
            parts = fn.split('_')
            if parts[-1] == 'RPE':
                return parts[-2]
        except Exception:
            pass
        return None
#

def iter_rpe_stacks(root_dir, recurse=1):
    subdirs = []
    for fn in os.listdir(root_dir):
        fpath = os.path.join(root_dir, fn)
        if os.path.isdir(fpath):
            subdirs.append(fpath)
            continue
        if fn.lower().endswith('.rpe.json'):
            yield fpath
    if recurse > 0:
        for cdir in subdirs:
            for fpath in iter_rpe_stacks(cdir, recurse=recurse-1):
                yield fpath
#

if __name__ == '__main__':
    
    fdir = r'C:\RPE_Data\Z01'
    
    import skimage
    import scipy
    import torch
    
    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
    DEFAULT_DATA_DIR = ROOT_DIR
    if not ROOT_DIR in sys.path:
        sys.path.append(ROOT_DIR)  # To find local version of the library
    
    from mrcnn_torch.mrcnn_model import build_model, predict_one
    from tiletools import slice_area
    from savgol import SavitzkyGolay2D
    from tm_config import TM_Config
    
    import imagetools
    
    def segment_stack(fpath, cfg, model, device):
        fstk = RpeStack(fpath)
        chname = cfg.class_name
        print(f'Segmenting {chname} of {fstk.name}')
        norm_map = {}
        if cfg.image_type == 'RGB':
            norm_map = fstk._rgb_norm_map()
        else:
            otsu = fstk.getChannelOtsu(chname)
            sc_norm = 16383./otsu
        otsu3 = 16383.
        if cfg.postproc & imagetools.POSTPROC_DNA:
            otsu3 = fstk.getChannelOtsu3(chname)[0]
        #
        tiles = slice_area((fstk.height, fstk.width), (cfg.tilesize, cfg.tilesize))
        m_frames = np.zeros(shape=(fstk.n_frames, fstk.height, fstk.width), dtype=np.uint8)
        #
        particles_3d = []
        for cframe in range(fstk.n_frames):
            print(f'  frame {cframe+1} of {fstk.n_frames}')
            particles_2d = []
            #
            if cfg.image_type == 'RGB':
                fr_data = fstk.getRgbFrame(cframe)
            else:
                fr_data = fstk.getFrame(cframe, chname).astype(np.float32)
                #
                if cfg.postproc & imagetools.POSTPROC_DNA:
                    fr_smooth = scipy.signal.convolve2d(fr_data, cfg.smooth_kern, boundary='symm', mode='same')
                    m_frames[cframe][fr_smooth > otsu3] = 0xFF
                    del fr_smooth
                #
                fr_data = fr_data * sc_norm
                fr_data[fr_data > 65530.] = 65530.
                fr_data = fr_data.astype(np.uint16)
            #
            ndimgs = []
            origs = []
            for x0, x1, y0, y1 in tiles:
                if cfg.image_type == 'RGB':
                    ndimg = fr_data[y0:y1+1, x0:x1+1, :]
                else:
                    ndimg = skimage.color.gray2rgb(fr_data[y0:y1+1, x0:x1+1])
                masks, scores = predict_one(model, device, ndimg, prob_thresh=cfg.prob_thresh)
                if len(masks.shape) != 3:
                    nmasks = np.empty(shape=(1, cfg.tilesize, cfg.tilesize), dtype=np.uint8)
                    nmasks[0] = masks
                    masks = nmasks
                masks[masks!=0] = 0xFF
                particles = imagetools.masks_to_particles(masks, x0, y0)
                particles_2d.extend(particles)
            #
            particles_3d.append(particles_2d)
        #
        tdir = fstk.subdir(cfg.subdir)
        os.makedirs(tdir, exist_ok=True)
        csvname = os.path.join(tdir, fstk.tname(chname, '.csv'))
        tifname = os.path.join(tdir, fstk.tname(chname, '.tif'))
        print('3D assembly...')
        imagetools.assemble_ml(particles_3d, m_frames, csvname, cfg.postproc, cfg.good_iou, cfg.ok_iou)
        print('Write:', tifname)
        imwrite(tifname, m_frames, photometric='minisblack')
    #

    
    cfg = TM_Config(ROOT_DIR, 'DNA', 'BW')
    cfg.postproc = imagetools.POSTPROC_DNA

    cfg.WEIGHTS_SUBDIR = os.path.join(ROOT_DIR, 'model_weights')
    cfg.tilesize = 768
    cfg.score_thresh = 0.65
    cfg.prob_thresh = 0.7
    
    cfg.sg = SavitzkyGolay2D(5, 2)
    cfg.smooth_kern = cfg.sg.kernel(0, 0)
    
    cfg.subdir = 'Predicted'
    cfg.good_iou = 0.6
    cfg.ok_iou = 0.5
    
    epoch, mwpath = cfg.find_model_weights(cfg.WEIGHTS_SUBDIR)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = build_model(
        num_classes=cfg.num_classes,
        detections_per_img=cfg.detections_per_img,
        score_thresh=cfg.score_thresh)
    model.to(device)
    
    model.load_state_dict(torch.load(mwpath, map_location=device))
    model.eval()
    print(f'Created Mask_RCNN instance, device "{device}", weights from "{os.path.basename(mwpath)}"')

    for fpath in iter_rpe_stacks(fdir):
        segment_stack(fpath, cfg, model, device)
    
    print('')
    print('All Done.')
    sys.exit(0)
    
    
    