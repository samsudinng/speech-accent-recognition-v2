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

### feature files
FPATH="/storage/sa0002ng/features_logspec200_new/"

### version
###     v1: training image segmented into 3sec segments, non-overlapping
###         organize according to torchvision ImageFolder datasets folder organization 
###
###     v2: same as v1, but training image has additional non-segmented spectrogram image
VERSION='v1' 

### paths to the output .png images
ENABLE_FEATURE2PNG=true
TRAINIMGPATH="/storage/sa0002ng/features_logspec200_new/train_img_"$VERSION"/"
DEVIMGPATH="/storage/sa0002ng/features_logspec200_new/dev_img_"$VERSION"/"
TESTIMGPATH="/storage/sa0002ng/features_logspec200_new/test_img_"$VERSION"/"


###########################################
### SOME DETAILED SETTINGS
### *you probably won't need to touch these                
###########################################

### path to the metadata files
METAPATH="metadata/"
FEATURE=logspec200
SEGMENTSIZE=300
### split the data set into chuncks
N_TRAIN_SPLIT=8
N_DEV_SPLIT=1
N_TEST_SPLIT=1

##Set to 1 to extract train/dev and test dataset
XTRACT_TRAIN=1
XTRACT_TEST=1

## set to 1 to delete wav files after features extraction
TO_DELETE_FEAT=0


###########################################
### CONVERT FEATURES TO PNG IMAGES
###########################################

if [ "$ENABLE_FEATURE2PNG" = true ]; then

    OUTPATH=$TRAINIMGPATH
    if [ -d $OUTPATH ]; then
        echo "resetting "$OUTPATH
        rm -r $OUTPATH
    fi
    echo "Creating "$OUTPATH" directory"
    mkdir $OUTPATH
    for i in {0..7}; do
        mkdir $OUTPATH/$i;
    done



    OUTPATH=$DEVIMGPATH
    if [ -d $OUTPATH ]; then
        echo "resetting "$OUTPATH
        rm -r $OUTPATH
    fi
    echo "Creating "$OUTPATH" directory"
    mkdir $OUTPATH


    OUTPATH=$TESTIMGPATH
    if [ -d $OUTPATH ]; then
        echo "resetting "$OUTPATH
        rm -r $OUTPATH
    fi
    echo "Creating "$OUTPATH" directory"
    mkdir $OUTPATH

    echo "Converting features to .png images"
    python features_to_png.py \
        --feature $FEATURE \
        --trainpath $TRAINIMGPATH \
        --version $VERSION \
        --devpath $DEVIMGPATH \
        --testpath $TESTIMGPATH \
        --fpath $FPATH \
        --metapath $METAPATH \
        --segmentsize $SEGMENTSIZE \
        --trainsplit $N_TRAIN_SPLIT \
        --devsplit $N_DEV_SPLIT \
        --testsplit $N_TEST_SPLIT \
        --xtrain $XTRACT_TRAIN \
        --xtest $XTRACT_TEST \
        --delete $TO_DELETE_FEAT
fi
