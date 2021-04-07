# Speech Accent Recognition with Spectrogram Images
Classifying accented English speech with image classification techniques


## Requirement: 

- Python 3.7.10

## Usage:

### 1. Clone the repository, create virtual environment and install dependencies

```
git clone https://github.com/samsudinng/speech-accent-recognition-v2.git
cd speech-accent-recognition-v2
python -m venv env-name
source env-name/bin/activate
pip install -r requirements.txt
```

### 2. Extract spectrogram images features

```
cd features_extraction
source wav_to_features.sh
source features_to_png.sh
```

The following variables should be set in the above scripts accordingly.

| **Script**        | **Variable**           | **Remark**  |
|:------------- |:-------------|:-----|
| **wav_to_features.sh**      | `VENV` | path to virtual environment created in step 1 (path-to/env-name) |
|       | `TRAINWAVPATH`, `TESTWAVPATH`| path to AESRC2020 Accent160 train/dev and test .wav files  |
|  | `FPATH`      | path to save the output spectrogram features files (.pkl)    |
| **features_to_png.sh** | `VENV` | same as above |
|  | `FPATH` | same as above |
|  | `TRAINIMGPATH`, `DEVIMGPATH`, `TESTIMGPATH` | path to save the train/dev/test spectrogram images (.png) |
|  | `VERSION` | `v1` : training spectrogram images are segmented into 3-sec, non-overlapping image segments <br> `v2` : same as `v1` + one 3-sec segment from position 1.5-4.5 sec + one center-cropped 3-sec segment|

### 3. Train/Dev/Test

To train (including dev and test per epoch):

```
python train_spectrogram_imagenet.py --logdir path-to/logs --config config.json
```

To resume from epoch #8:

```
python train_spectrogram_imagenet.py --resume path-to/logs/checkpoint_spectrogram_epoch8.pth --logdir path-to/logs_resumed --config config.json
```

Model training can be configured from `config.json`. Several preset configuration files are provided in config/ folder. For details on the format of configuration file, see the explanation at the end of this readme.

### 4. Logging/checkpoints/results

Checkpoint
- `path-to/logs/checkpoint_spectrogram_epochX.pth` : checkpoint, only the last two training epochs are kept

Best model (highest dev accuracy)
- `path-to/logs/bestmodel_epochN.pth`

Results log (train/dev/test accuracies, test confusion matrix)
- `path-to/logs/log.file`

### 5. Result monitoring (tensorboard)
By default, tensorboard logging is enabled. The event file can be found in `path-to/logs/`.


## Configuration file

The configuration file (.json) is in the format of Python dictionary. The details are as follows.

```
{

# For model, optimizer, loss and learning rate scheduler, the value should be Python code used to instantiate these
#     classes. In the code, the classes will be instantiated from these values using eval() function. 

"model"         : "AlexNetGAP()",
"optimizer"     : "torch.optim.AdamW(model.parameters(), lr = 1e-5)",
"loss"          : "torch.nn.CrossEntropyLoss()",
"scheduler"     : "optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, mode='max')",


# Set batchsize, number of epochs and probability of applying spectral augmentation (frequency/time masking)

"batchsize"     : 196,
"epochs"        : 30,
"p_specaugment" : 0.6,


# Set path to the train, dev and test input images. 

"trainpath"     : "/storage/sa0002ng/features_logspec200_new/train_img_v2/",
"devpath"	: "/storage/sa0002ng/features_logspec200_new/dev_img_v2/",
"testpath"	: "/storage/sa0002ng/features_logspec200_new/test_img_v2/",


# Metadata path can be kept unchanged as it resides within this repository. However, if this config file is not located
#       in the root folder of this repository, modify the relative/absolute path accordingly.

"mpath"         : "features_extraction/metadata/",

}
```
