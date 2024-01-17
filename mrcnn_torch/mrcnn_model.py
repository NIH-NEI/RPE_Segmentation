import os, sys

import numpy as np
import torch
import torch.utils.data

# https://github.com/pytorch/vision/blob/main/torchvision/models/detection/backbone_utils.py
# https://github.com/pytorch/vision/blob/main/torchvision/models/detection/mask_rcnn.py

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor

from . import utils, engine
from .mrcnn_dataset import MaskRCNNDataset

def build_model(pretrained=True, num_classes=2, detections_per_img=2048, trainable_layers=5,
        score_thresh=0.25, hidden_layer=256):
    # resnet101
    bkb_weights = 'ResNet101_Weights.DEFAULT' if pretrained else None
    backbone = resnet_fpn_backbone(backbone_name='resnet101', weights=bkb_weights,
            trainable_layers=trainable_layers)
    
    # resnext101_32x8d
    # bkb_weights = 'ResNeXt101_32X8D_Weights.DEFAULT' if pretrained else None
    # backbone = resnet_fpn_backbone(backbone_name='resnext101_32x8d', weights=bkb_weights,
    #         trainable_layers=trainable_layers)
    model = MaskRCNN(backbone, num_classes,
            box_detections_per_img=detections_per_img,
            box_score_thresh=score_thresh,
            min_size=1024, max_size=1280)
    #in_features = model.roi_heads.box_predictor.cls_score.in_features
    #model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    #model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model
#

def predict_one(model, device, ndimg, prob_thresh=0.5):
    img = MaskRCNNDataset.ndimg_to_torch(ndimg)
    img = img.to(device)
    pred = model([img])[0]
    masks = (pred['masks'] > prob_thresh).squeeze().detach().cpu().numpy().astype(np.uint8)
    scores = pred['scores'].squeeze().detach().cpu().numpy()
    return masks, scores
#

def predict_batch(model, device, ndimgs, prob_thresh=0.5):
    imgs = []
    for ndimg in ndimgs:
        img = MaskRCNNDataset.ndimg_to_torch(ndimg)
        img = img.to(device)
        imgs.append(img)
    preds = model(imgs)
    res = []
    for pred in preds:
        masks = (pred['masks'] > prob_thresh).squeeze().detach().cpu().numpy().astype(np.uint8)
        scores = pred['scores'].squeeze().detach().cpu().numpy()
        res.append((masks, scores))
    return res
#

def train(model, device, cfg, ndds, weights_dir, augm=None, num_epochs=10):
    os.makedirs(weights_dir, exist_ok=True)
    #
    epoch, wpath = cfg.find_model_weights(weights_dir)
    if not wpath is None:
        start_epoch = epoch + 1
        print('Loading model weights from:', wpath)
        model.load_state_dict(torch.load(wpath))
    else:
        start_epoch = 1
    end_epoch = start_epoch + num_epochs - 1
    #
    n_items = len(ndds)
    n_train = int(n_items * cfg.train_val_split)
    train_ds = MaskRCNNDataset(ndds, idxrng=range(0, n_train), augm=augm)
    val_ds = MaskRCNNDataset(ndds, idxrng=range(n_train, n_items), augm=None)
    print('Training items:', len(train_ds), ': Validation items:', len(val_ds))

    # Data loaders
    data_loader = torch.utils.data.DataLoader(train_ds,
                batch_size=cfg.batch_size, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(val_ds,
                batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
    
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    
    # and a learning rate scheduler which scales the LR by 'gamma' every 'step_size' epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    
    for epoch in range(start_epoch, end_epoch+1):
        if hasattr(ndds, 'shuffle'):
            ndds.shuffle()

        # train for one epoch, printing every 10 iterations
        engine.train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        engine.evaluate(model, data_loader_test, device=device)
        
        wfn = cfg.weights_name(epoch)
        wpath = os.path.join(weights_dir, wfn)
        print('Save model weights to:', wpath)
        torch.save(model.state_dict(), wpath)
    #

