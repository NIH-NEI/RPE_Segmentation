# RPE_Segmentation

Machine Learning suite for training Pytorch/torchvision based Mask_RCNN model to perform
instance segmentation of biomedical images.

*Andrei Volkov (NEI/NIH via MSC)*

## Setting Up Development Environment

1. Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
or [Anaconda](https://www.anaconda.com/products/individual). On Windows systems you will
also need to install C++ build tools, such as Microsoft Visual C++ or MinGW. When installing MSVC,
make sure you have checked the box for x64/x86 build tools, such as
`MSVC v143 - VS 2022 C++ x64/x86 build tools (latest)`.

2. Check out **RPE_Segmentation** to a local directory `<prefix>/RPE_Segmentation`.
(Replace `<prefix>` with any suitable local directory).

3. Run Anaconda Prompt (or Terminal), cd to `<prefix>/RPE_Segmentation`.

4. Create Conda Virtual Environment (do this once on the first run):

	`conda env create --file conda-environment.yml`
   
5. Activate the Virtual Environment:

	`conda activate RPE_Segmentation`
	
6. Install extra packages:

	`pip install -r x-requirements.txt`
   
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

1. Download [model_weights_DNA_BW.zip](https://github.com/NIH-NEI/RPE_Segmentation/releases/download/pretrained/model_weights_DNA_BW.zip),
[model_weights_Actin_RGB.zip](https://github.com/NIH-NEI/RPE_Segmentation/releases/download/pretrained/model_weights_Actin_RGB.zip)
and [model_weights_Actin_BW.zip](https://github.com/NIH-NEI/RPE_Segmentation/releases/download/pretrained/model_weights_Actin_BW.zip), then
unzip them into `<dataprefix>` (this will create a sub-directory `<dataprefix>/model_weights` and place *.pth
files there).

2. Download [Test_Data.zip](https://github.com/NIH-NEI/RPE_Segmentation/releases/download/testdata/Test_Data.zip)
and unzip it into `<dataprefix>` (this will create a sub-directory `<dataprefix>/Test_Data`
and place some input data there).

3. At conda prompt, activate *RPE_Segmentation* and cd to `<prefix>/RPE_Segmentation`, then type:

`python predict.py -d <dataprefix> All Test_Data`

(replace `<dataprefix>` with the actual directory).

Results can be found in `<dataprefix>/Test_Data/Predicted`.

## Training models.

1. Download [RPE_Training_DNA_BW.zip](https://github.com/NIH-NEI/RPE_Segmentation/releases/download/training/RPE_Training_DNA_BW.zip)
and [RPE_Training_Actin_RGB.zip](https://github.com/NIH-NEI/RPE_Segmentation/releases/download/training/RPE_Training_Actin_RGB.zip),
then unzip them into `<dataprefix>`.
The actual training data will be found in `<dataprefix>/RPE_Training/Mask_RCNN/DNA` and
<dataprefix>/RPE_Training/Mask_RCNN/Actin`. *Note that this training data is for demo purposes only, since
the full training data set is very large. It is available upon request.*

2. At conda prompt, activate *RPE_Segmentation* and cd to `<prefix>/RPE_Segmentation`, then type:

`python train.py -d <dataprefix> DNA`

`python train.py -d <dataprefix> Actin`

(replace `<dataprefix>` with the actual directory).

The training script will look for the latest epoch of an existing model in `<dataprefix>/model_weights` and
write new model weight files into the same directory with epoch number incremented after each epoch. In addition, loss metrics
(averaged across each epoch) are stored in `<dataprefix>/logs/*-loss.csv` files, and validation metrics (calculated after
each epoch), in `<dataprefix>/logs/*-iou.csv` files.

## Triaining models with your own data

To train models with your own data, prepare the training data in a directory structured as follows:

```
C:\RPEMapDataRoot\RPE_Training
└───Mask_RCNN
    ├───Actin
    │       P1-W1-TOM_E02_F001-000-000.png
    │       P1-W1-TOM_E02_F001-000-001.png
    │       P1-W1-TOM_E02_F001-000-002.png
    │       ...
    │       P1-W1-TOM_E02_F001_annotations_via.json
    │
    │       P1-W1-ZO1_D02_F001-000-000.png
    │       P1-W1-ZO1_D02_F001-000-001.png
    │       P1-W1-ZO1_D02_F001-000-002.png
    │       ...
    │       P1-W1-ZO1_D02_F001_annotations_via.json
    │       ..............................
    ├───DNA
    │       P1-W1-TOM_E02_F001-000-000.png
    │       P1-W1-TOM_E02_F001-000-001.png
    │       P1-W1-TOM_E02_F001-000-002.png
    │       ...
    │       P1-W1-TOM_E02_F001_annotations_via.json
    │       ..............................
```

The [prefix]-ZZZ-NNN.png files contain the source images, the [prefix]_annotations_via.json files
(formatted in VIA-compatible way) contain annotations (ground truth).

## Evaluating segmentation quality

The test data [Test_Data.zip](https://github.com/NIH-NEI/RPE_Segmentation/releases/download/testdata/Test_Data.zip)
comes with human-created annotations, which can be used to evaluate the segmentation quality.
Follow steps in [Training models](#training-models) section, but instead of executing `predict.py`
type the following command:

`python evaluate.py <dataprefix>/Test_Data [-f]`

Use `-f` option to re-do previous segmentation, e.g. after updating Mask_RCNN model weights. For list of all available
command-line options, type:

`python evaluate.py -h`

Evaluation results are stored in `<dataprefix>/Test_Data/EvaluationResults`. Comparison is done separately for Actin
and DNA channels in each stack. The result is a plain text file formatted like this:

```
+-------------------------------+---------+----------+
| 2D Comparison - Actin of P1-W2-ZO1_D02_F006        |
+-------------------------------+---------+----------+
| Primary annotations           |    8206 |  100.00% |
| Secondary annotations         |    7565 |   92.19% |
| Fragmented                    |       0 |    0.00% |
| Fused                         |      21 |    0.26% |
| False Positives               |     174 |    2.12% |
| False Negatives               |     792 |    9.65% |
+-------------------------------+---------+----------+
| Matches at IoU >= 95%         |    1462 |   17.82% |
| Matches at IoU >= 90%         |    5662 |   69.00% |
| Matches at IoU >= 80%         |    7031 |   85.68% |
| Matches at IoU >= 75%         |    7139 |   87.00% |
| Matches at IoU >= 50%         |    7309 |   89.07% |
+-------------------------------+---------+----------+

+-------------------------------+---------+----------+
| 3D Comparison - Actin of P1-W2-ZO1_D02_F006        |
+-------------------------------+---------+----------+
| Primary annotations           |     607 |  100.00% |
| Secondary annotations         |     575 |   94.73% |
| Fragmented                    |       0 |    0.00% |
| Fused                         |      18 |    2.97% |
| False Positives               |       2 |    0.33% |
| False Negatives               |      27 |    4.45% |
+-------------------------------+---------+----------+
| Matches at IoU >= 95%         |       8 |    1.32% |
| Matches at IoU >= 90%         |     141 |   23.23% |
| Matches at IoU >= 80%         |     365 |   60.13% |
| Matches at IoU >= 75%         |     417 |   68.70% |
| Matches at IoU >= 50%         |     512 |   84.35% |
+-------------------------------+---------+----------+
```

Comparison is done in both 2D, whivch is good for evaluating quality of raw Mask_RCNN segmentations, and 3D - for evaluating
overall quality of the whole process, including 3D assembly. The metrics presented are:

- *Primary annotations*: total number of annotations in the primary (manual) set

- *Secondary annotations*: total number of annotations in the secondary (automatic) set

- *Fragmented*: number of mismatches due to fragmentation, i.e. an annotation from the manual set overlaps with 2 or more
annotations from the automatic set

- *Fused*: number of mismatches due to "fusions", i.e. 2 or more annotations from the manual set overlaps with one annotation
from the automatic set

- *False Positives*: number of false positives, annotations from the automatic set not matching anything in the manual set

- *False Negatives*: number of false negatives, annotations from the manual set not matching any automatic annotations

- *Matches at IoU >= N%*: Number of 1-to-1 matches having Intersection over Union value greater or equal to the specified value

The percentages are given relative to the *Primary annotations*.

