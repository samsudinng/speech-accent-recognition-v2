#!/bin/sh
#SBATCH --partition=SCSEGPU_MSAI
#SBATCH --qos=q_msai
#SBATCH --mem=8000M
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --job-name=TestJob
#SBATCH --output=test.out
#SBATCH --error=testError.err


VENV="/home/MSAI/sa0002ng/ACCENT/spectrogram/speech_accent_recognition_new/venv/accent_new_venv"
source $VENV/bin/activate


##########################################
### SET THESE PATHS TO YOUR OWN
##########################################

### paths to the .wav and output feature files
ENABLE_WAV2FEATURE=true
TRAINWAVPATH="/storage/sa0002ng/accent_audio/"
TESTWAVPATH="/storage/sa0002ng/accent_testaudio/"
FPATH="/storage/sa0002ng/features_logspec200_new/"


###########################################
### SOME DETAILED SETTINGS
### *you probably won't need to touch these                
###########################################

### path to the metadata files
METAPATH="metadata/"

FEATURE=logspec200
ZSCALERFILE="None"

# test
### split the data set into chuncks
N_TRAIN_SPLIT=8
N_DEV_SPLIT=1
N_TEST_SPLIT=1

##Set to 1 to extract train/dev and test dataset
XTRACT_TRAIN=1
XTRACT_TEST=1

## set to 1 to delete wav files after features extraction
TO_DELETE_WAV=0

###########################################
### CONVERT .WAV TO FEATURES
###########################################

if [ "$ENABLE_WAV2FEATURE" = true ]; then
    if [ -d $FPATH ]; then
        echo $FPATH" already exists, resetting the directory"
        rm -r $FPATH
    fi
    echo "Creating "$FPATH" directory"
    mkdir $FPATH

    echo "Converting .wav to features"
    python wav_to_features.py \
        --feature $FEATURE \
        --trainpath $TRAINWAVPATH \
        --testpath $TESTWAVPATH \
        --fpath $FPATH \
        --zscore $ZSCALERFILE \
        --metapath $METAPATH \
        --trainsplit $N_TRAIN_SPLIT \
        --devsplit $N_DEV_SPLIT \
        --testsplit $N_TEST_SPLIT \
        --xtrain $XTRACT_TRAIN \
        --xtest $XTRACT_TEST \
        --delete $TO_DELETE_WAV
fi

