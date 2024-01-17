# RPE_Segmentation

Machine Learning suite for training Pytorch/torchvision based Mask_RCNN model to perform
instance segmentation of biomedical images.

*Andrei Volkov (NEI/NIH via MSC)*

## Setting Up Development Environment

1. Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
or [Anaconda](https://www.anaconda.com/products/individual).

2. Check out **RPE_Segmentation** to a local directory `<prefix>/RPE_Segmentation`.
(Replace `<prefix>` with any suitable local directory).

3. Run Anaconda Prompt (or Terminal), cd to `<prefix>/RPE_Segmentation`.

4. Create Conda Virtual Environment (do this once on the first run):

	`conda env create --file conda-environment.yml`
   
5. Activate the Virtual Environment:

	`conda activate RPE_Segmentation`
	
6. Build the native extension (do this once on the first run or after making changes to the native extension):

	`pip install ./imagetools`
   
To delete the Virtual environment at the Conda prompt, deactivate it first, if it is active:

`conda deactivate`

then type:

`conda remove --name RPE_Segmentation`


**Note:** Both prediction and training scripts (`predict.py` and `train.py`) need some auxiliary data, such
as trained model weights, training data, etc. It is highly recommended to create a separate directory for
this purpose, we will be referring to it as `<dataprefix>'. Use the `--data-dir <dataprefix>` command option to
instruct the scripts to look for auxiliary data in this directory. By default the scripts will look for data
in the same directory as where they are located.

## Using pre-trained models for predictions

1. Download `model_weights_DNA_BW.zip`, `model_weights_Actin_RGB.zip` and `model_weights_Actin_BW.zip`, then
unzip them into `<dataprefix>` (this will create a sub-directory `<dataprefix>/model_weights` and place *.pth
files there).

2. Download `Test_Data.zip` and unzip it into `<dataprefix>` (this will create a sub-directory `<dataprefix>/Test_Data`
and place some input data there).

3. At conda prompt, activate *RPE_Segmentation* and cd to `<prefix>/RPE_Segmentation`, then type:

`python predict.py -d <dataprefix> All Test_Data`

(replace `<dataprefix>` with the actual directory).

Results can be found in `<dataprefix>/Test_Data/W1/Predicted` and `<dataprefix>/Test_Data/W2/Predicted`.

## Training models.

1. Download `RPE_Training_DNA_BW.zip` and `RPE_Training_Actin_RGB.zip`, then unzip them into `<dataprefix>`.
The actual training data will be found in `<dataprefix>/RPE_Training/Mask_RCNN/DNA` and
<dataprefix>/RPE_Training/Mask_RCNN/Actin`. Note that this training data is for demo purposes only, since
the full training data set is very large. It is available upon request.

2. At conda prompt, activate *RPE_Segmentation* and cd to `<prefix>/RPE_Segmentation`, then type:

`python train.py -d <dataprefix> DNA`

`python train.py -d <dataprefix> Actin`

(replace `<dataprefix>` with the actual directory).

The training script will look for the latest epoch of an existing model in `<dataprefix>/model_weights` and
write new model weight files into the same directory with epoch number incremented after each epoch.

