# Speech Accent Recognition with Spectrogram Images
Speech accent recognition with image classification technique


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

### 3. Train/Dev/Test

To train (including dev and test per epoch):

```
python train_spectrogram_imagenet.py --logdir path-to/logs --config config.json
```

To resume from epoch #8:

```
python train_spectrogram_imagenet.py --resume path-to/logs/checkpoint_spectrogram_epoch8.pth --logdir path-to/logs_resumed --config config.json
```

### 4. Logging/ Results

Checkpoint
- `path-to/logs/checkpoint_spectrogram_epochX.pth` : checkpoint, only the last two training epochs are kept

Best model (highest dev accuracy)
- `path-to/logs/bestmodel_epochN.pth`

Results log (train/dev/test accuracies, test confusion matrix)
- `path-to/logs/log.file`

### 5. Result monitoring (tensorboard)
By default, tensorboard logging is enabled. The event file can be found in `path-to/logs/`.
