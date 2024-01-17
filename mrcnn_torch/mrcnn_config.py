import os

class MaskRCNNBaseConfig:
    model_name = 'Mask_RCNN'

    # such as 'Actin', 'DNA', 'Z01', etc.
    class_name = 'YourClassName'

    # Source image type, 'BW' for monochrome or 'RGB' for color
    image_type = 'BW'

    # Background + Foreground
    num_classes = 2
    
    # Disable GPU if False
    use_gpu = True

    # Number of pairs (source image + target) per training step
    batch_size = 1

    # Training/Validation split (0.01 .. 0.99)
    train_val_split = 0.9
    
    # Number of trainable layers in the backbone (0..5)
    trainable_layers = 5
    
    # Initial learning rate, momentum and weight decay for SGD optimizer
    learning_rate = 0.0005
    momentum = 0.9
    weight_decay = 0.0005
    
    # Learning rate scheduler: take learning rate 'gamma' times after each 'step_size' epochs
    step_size = 10
    gamma = 0.1
    
    # Max. box detections per image
    detections_per_img = 2048

    # Return only masks with the score above this value
    score_thresh = 0.5

    # Foreground/Background threshold: set pixel to foreground when predicted probability is above this value
    prob_thresh = 0.65
    #
    def weights_name(self, epoch):
        return f'{self.model_name}-{self.class_name}-{self.image_type}-{epoch:04d}.pth'
    #
    def find_model_weights(self, weights_dir):
        epoch = 0
        wpath = None
        prefix = f'{self.model_name}-{self.class_name}-{self.image_type}-'
        for fn in os.listdir(weights_dir):
            if not fn.startswith(prefix): continue
            bn, ext = os.path.splitext(fn)
            if ext.lower() != '.pth': continue
            try:
                e = int(bn.split('-')[-1])
                if e > epoch:
                    p = os.path.join(weights_dir, fn)
                    if os.path.isfile(p):
                        epoch = e
                        wpath = p
            except Exception:
                pass
        #
        return epoch, wpath
    #


