# Speech Accent Recognition with Spectrogram Images
Classifying accented English speech with: 
- Imagenet pre-trained image classifier
- Finetuning on [Accent160](https://m.datatang.com/shujutang/static/file/AESRC2020.pdf) dataset (speech features are converted to .png images for classification)

**Features extraction (.wav to .png images)**

- [x] Log-spectrogram -> 200 x t pixels (t = 10 msec resolution)
- [x] [wav2vec](https://arxiv.org/abs/1904.05862) -> 512 x t pixels (t = 10 msec resolution)
- [ ] [wav2vec2](https://arxiv.org/abs/2006.11477)

**Accent classifiers (Imagenet pre-trained models from torchvision.models with 8 output classes)**
- [x] AlexNetGAP      -> AlexNet with linear classifier replaced by global average pooling (GAP)
- [x] VGG16BnGAP      -> VGG16 with batchnorm and GAP
- [x] VGG19BnGAP      -> VGG19 with batchnorm and GAP
- [x] Resnet18
- [x] Resnet34
- [x] Resnet50    

**Model training and evaluation**
- [x] Configurable via .json config file
- [x] At each epoch: train on training images; evaluate train, dev and test accuracy at the end of each epoch
- [x] Tensorboard monitoring
- [x] Training resume from checkpoint (only the last two training checkpoints are saved)
- [x] Best model (.pth) selected and saved based on best dev accuracy   

**Requirements** 

- [x] Python 3.7.10
- [x] Required dependencies in `requirements.txt`, can be installed wit `pip`  

---

## Usage:

### 1. Clone the repository, create virtual environment and install dependencies

```python
git clone https://github.com/samsudinng/speech-accent-recognition-v2.git
cd speech-accent-recognition-v2
python -m venv env-name
source env-name/bin/activate
pip install -r requirements.txt
```


### 2. Extract speech feature images

#### 2.1 To extract log-spectrogram images:

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


#### 2.2 To extract wav2vec images:

```
cd features_extraction/wav2vec
source wav_to_wav2vec.sh
```

The following variables should be set in the above scripts accordingly.

| **Script**        | **Variable**           | **Remark**  |
|:------------- |:-------------|:-----|
| **wav_to_wav2vec.sh**      | `VENV` | path to virtual environment created in step 1 (path-to/env-name) |
|       | `TRAINWAVPATH`, `TESTWAVPATH`| path to AESRC2020 Accent160 train/dev and test .wav files  |
|  | `TRAINIMGPATH`, `DEVIMGPATH`, `TESTIMGPATH` | path to save the train/dev/test spectrogram images (.png) |



### 3. Train/Dev/Test

To train (including dev and test per epoch):

```
python train_accent_imagenet.py --logdir path-to/logs --config config/config_alexnet-196-m5.json
```

To resume from epoch #8:

```
python train_accent_imagenet.py --resume path-to/logs/checkpoint_spectrogram_epoch8.pth --logdir path-to/logs_resumed --config config/config_alexnet-196-m5.json
```

Model training can be configured from .json config file. Several preset configuration files are provided in config/ folder. For details on the format of configuration file, see the explanation at the end of this readme.

### 4. Logging/checkpoints/results

Checkpoint
- `path-to/logs/checkpoint_spectrogram_epochX.pth` : checkpoint, only the last two training epochs are kept

Best model (highest dev accuracy)
- `path-to/logs/bestmodel_epochN.pth`

Results log (train/dev/test accuracies, test confusion matrix)
- `path-to/logs/log.file`

### 5. Result monitoring (tensorboard)
By default, tensorboard logging is enabled. The event file can be found in `path-to/logs/`.

---

## Configuration file

The configuration file (.json) is in the format of Python dictionary ("key":"value" pairs). The details are as follows.

```
{

# MODEL & TRAINING
# For model, optimizer, loss and learning rate scheduler, the value should be a string of 
#     Python code used to instantiate these classes. In the code, the classes will be  
#     instantiated from these string values using eval() function. 

"model"         : "AlexNetGAP()",
"optimizer"     : "torch.optim.AdamW(model.parameters(), lr = 1e-5)",
"loss"          : "torch.nn.CrossEntropyLoss()",
"scheduler"     : "optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, mode='max')",
"epochs"        : 30,


# DATALOADING
# "feature" is used to configure the image pre-transform during data loading. 
#     Set to "spectrogram" or "wav2vec" according to the feature images used for training.

"feature"       : "wav2vec",
"batchsize"     : 196,
"num_workers"   : 0,


# AUGMENTATION
# Set batchsize, number of epochs and probability of applying spectral augmentation 
"p_specaugment" : 0.6,


# PATHS
# Set path to the train, dev and test input images. Metadata path can be kept unchanged 
#     as it resides within this repository.

"trainpath"     : "/storage/sa0002ng/features_logspec200_new/train_img_v2/",
"devpath"	: "/storage/sa0002ng/features_logspec200_new/dev_img_v2/",
"testpath"	: "/storage/sa0002ng/features_logspec200_new/test_img_v2/",
"mpath"         : "features_extraction/metadata/",

}
```
