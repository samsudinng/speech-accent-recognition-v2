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
TRAINWAVPATH="/storage/sa0002ng/accent_audio/"
TESTWAVPATH="/storage/sa0002ng/accent_testaudio/"
TRAINIMGPATH="/storage/sa0002ng/features_wav2vec/train_img/"
DEVIMGPATH="/storage/sa0002ng/features_wav2vec/dev_img/"
TESTIMGPATH="/storage/sa0002ng/features_wav2vec/test_img/"


###########################################
### SOME DETAILED SETTINGS
### *you probably won't need to touch these                
###########################################

### path to the metadata files
METAPATH="../metadata/"


###########################################
### CONVERT .WAV TO WAV2VEC
###########################################

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


echo "Converting .wav to features"
python wav_to_wav2vec.py \
        --trainpath $TRAINWAVPATH \
        --testpath $TESTWAVPATH \
        --metapath $METAPATH \
        --trainimgpath $TRAINIMGPATH \
        --devimgpath $DEVIMGPATH \
        --testimgpath $TESTIMGPATH \


