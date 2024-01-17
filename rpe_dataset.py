import os, sys
import json
import random

import numpy as np
import imageio
import skimage.draw
import imgaug as ia
from imgaug import augmenters as iaa

try:
    imread = imageio.v2.imread
except Exception:
    imread = imageio.imread

class RPE_Dataset(object):
    def __init__(self, dataset_dir, min_ann_per_img=1, skip=None):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.min_ann_per_img = min_ann_per_img
        self.skip = skip
        #
        if self.skip:
            if isinstance(self.skip, str):
                self.skip = [skip,]
        else:
            self.skip = []
        #
        self.dataitems = []
        #
        self._read_dataset()
    #
    def _read_dataset(self):
        for fn in os.listdir(self.dataset_dir):
            if not fn.endswith('_annotations_via.json'):
                continue
            _skip = False
            for ss in self.skip:
                if ss in fn:
                    _skip = True
                    break
            if _skip: continue
            jpath = os.path.join(self.dataset_dir, fn)
            try:
                with open(jpath, 'r') as fi:
                    annotations = json.load(fi)
                for a in annotations.values():
                    if a['regions']:
                        self._load_data_item(a)
            except Exception as ex:
                print(ex)
                continue
    #
    def _load_data_item(self, a):
        if type(a['regions']) is dict:
            _polygons = [r['shape_attributes'] for r in a['regions'].values()]
        else:
            _polygons = [r['shape_attributes'] for r in a['regions']]
        if len(_polygons) < self.min_ann_per_img:
            return
        polygons = [(p['all_points_y'], p['all_points_x']) for p in _polygons]
        image_path = os.path.join(self.dataset_dir, a['filename'])
        if not os.path.isfile(image_path):
            return
        try:
            fa = a['file_attributes']
            height = int(fa['height'])
            width = int(fa['width'])
        except Exception:
            image = imread(image_path)
            height, width = image.shape[:2]
        self.dataitems.append({
            'imgpath': image_path,
            'width': width,
            'height': height,
            'polygons': polygons,
        })
    #
    def __len__(self):
        return len(self.dataitems)
    #
    def __getitem__(self, idx):
        item = self.dataitems[idx]
        img = self.load_image(item['imgpath'])
        masks = self.load_masks(item['width'], item['height'], item['polygons'])
        return img, masks
    #
    def shuffle(self):
        random.shuffle(self.dataitems)
    #
    def itemname(self, idx):
        imgpath = self.dataitems[idx]['imgpath']
        fn = os.path.basename(imgpath)
        bn, ext = os.path.splitext(fn)
        return bn
    #
    @staticmethod
    def load_image(imgpath):
        image = imread(imgpath)
        # If grayscale, convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        assert image.shape[2] in (3, 4), 'Image format not recognized'
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image
    #
    @staticmethod
    def load_masks(width, height, polygons):
        masks = []
        for i, (ys, xs) in enumerate(polygons):
            rr, cc = skimage.draw.polygon(ys, xs)
            mask = np.zeros(shape=(height, width), dtype=np.uint8)
            mask[rr, cc] = 1
            masks.append(mask)
        return masks
    #
    @staticmethod
    def masks_as_ndarray(masks):
        h, w = masks[0].shape
        omask = np.empty(shape=(len(masks), h, w), dtype=np.uint8)
        for i, mask in enumerate(masks):
            omask[i] = mask
        return omask
    #

class RPE_Augmenter(object):
    def __init__(self):
        # Image augmentation
        # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
        ia.seed(1)
        #
        aseq = iaa.Sequential([
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 30% of all images.
            iaa.Sometimes(
                0.3,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                rotate=(-25, 25), shear=(-7, 7)
            )
        ], random_order=True) # apply augmenters in random order
        #
        self.augmentation = iaa.SomeOf((0, 2), [
            iaa.OneOf([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270)
            ]),
            iaa.Sometimes(
                0.5,
                aseq
            ),
        ])
    #
    def augment(self, image, masks):
        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = self.augmentation.to_deterministic()
        image = det.augment_image(image)
        for i, mask in enumerate(masks):
            mask_shape = mask.shape
            masks[i] = det.augment_image(mask, hooks=ia.HooksImages(activator=hook))
            assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        #
        return image, masks
    #

