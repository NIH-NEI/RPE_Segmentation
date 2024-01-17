import numpy as np
import torch
import torch.utils.data

# The purpose of this class is to convert training data from numpy arrays to torch tensors,
# suitable for training MaskRCNN.
#
# Arguments:
# ndds - a list-like object, a class defining methods __len__() and __getitem__(),
#        containing elements as tuples (ndimg, masks);
#        ndimg is a numpy array shaped [H,W] or [H,W,C];
#        masks is a *non-empty* list of numpy arrays [H,W] of type uint8, containing GT masks.
#
# idxrng - a sequence or generator of indexes defining which subset of data items from 'ndds' to use.
#        if None, the whole ndds will be used to produce training data, idxrng=range(len(ndds)).
#
# augm - an augmentation function that takes (ndimg, masks) as parameters (same type as elements of 'ndds'),
#        and returns their modified versions. If None, no augmentation applied.
#
class MaskRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, ndds, idxrng=None, augm=None):
        self.ndds = ndds
        self.idxrng = idxrng if not idxrng is None else range(len(ndds))
        self.augm = augm
        #
        self.idxs = list(self.idxrng)
    #
    def __len__(self):
        return len(self.idxs)
    #
    def __getitem__(self, idx):
        for retry in range(3):
            # If masks are shifted away from the image as a result of augmentation, try 2 more times.
            _ndimg, _masks = self.ndds[self.idxs[idx]]
            ndimg, masks, bboxes = self._get_masks_and_boxes(_ndimg, _masks, self.augm)
            if len(bboxes) > 0: break
        else:
            # If 3 augmentation attempts failed to produce non-0 masks, give up on augmentation
            # and just return the original item
            _ndimg, _masks = self.ndds[self.idxs[idx]]
            ndimg, masks, bboxes = self._get_masks_and_boxes(_ndimg, _masks, None)
            assert len(bboxes) > 0, f'Image {idx}, no masks!'
        #
        num_objs = len(masks)
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks, dtype=np.uint8), dtype=torch.uint8)
        image_id = torch.tensor([idx])
        # Extra parameters to satisfy COCO API during validation
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        #
        target = {
            'image_id': image_id,
            'boxes': bboxes,
            'labels': labels,
            'masks': masks,
            'area': area,
            'iscrowd': iscrowd,
        }
        return self.ndimg_to_torch(ndimg), target
    #
    def _get_masks_and_boxes(self, ndimg, _masks, augm):
        if not augm is None:
            ndimg, _masks = augm(ndimg, _masks)
        masks = []
        bboxes = []
        for msk in _masks:
            try:
                bb = self.bounding_box(msk)
                masks.append(msk)
                bboxes.append(bb)
            except Exception:
                pass
        return ndimg, masks, bboxes
    #
    @staticmethod
    def ndimg_to_torch(ndimg):
        if ndimg.dtype == np.uint16:
            ndimg = ndimg.astype(np.float32) / 65535.
        else:
            ndimg = ndimg.astype(np.float32) / 255.
        ndimg = ndimg.transpose(2, 0, 1)
        return torch.as_tensor(ndimg, dtype=torch.float32)
    #
    # Throws an exception if mask is an empty mask (all 0-s)
    @staticmethod
    def bounding_box(mask):
        a = np.where(mask != 0)
        # (x, y, x+w, y+h)
        return np.min(a[1]), np.min(a[0]), np.max(a[1])+1, np.max(a[0])+1
    #

