import os, sys

from mrcnn_torch.mrcnn_config import MaskRCNNBaseConfig

class RPE_Config(MaskRCNNBaseConfig):
    RGB_CLASS_NAMES = ('Actin',)
    TRAINING_SUBDIR = 'RPE_Training'
    WEIGHTS_SUBDIR = 'model_weights'
    PREDICTIONS_SUBDIR = 'Predicted'
    tilesize = 768
    #
    # Postprocessing for 3D assembly: 0, imagetools.POSTPROC_DNA or imagetools.POSTPROC_ACTIN
    postproc = 0
    #
    # Minimum annotations per image (for acceptable training items)
    min_ann_per_img = 1
    #
    def __init__(self, data_dir, class_name, image_type=None):
        self.data_dir = data_dir
        self.class_name = class_name
        def_image_type = self.defaultImageType(self.class_name)
        self.image_type = image_type if image_type else def_image_type
        #
        self.is_default_image_type = self.image_type == def_image_type
        #
        self.tilesize = 768
    #
    def training_data_dir(self):
        tdir = os.path.join(self.data_dir, self.TRAINING_SUBDIR, self.model_name, f'{self.class_name}-{self.image_type}')
        if not os.path.isdir(tdir) and self.is_default_image_type:
            # Directory with explicit image type not found, try default if applicable
            _tdir = os.path.join(self.data_dir, self.TRAINING_SUBDIR, self.model_name, self.class_name)
            if os.path.isdir(_tdir):
                return _tdir
        return tdir
    #
    def model_weights_dir(self):
        return os.path.join(self.data_dir, self.WEIGHTS_SUBDIR)
    #
    @staticmethod
    def defaultImageType(class_name):
        if class_name in RPE_Config.RGB_CLASS_NAMES:
            return 'RGB'
        return 'BW'
    #
