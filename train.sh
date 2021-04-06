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

LOGDIR="train-imagenet"
LABEL="adamw-196-m5"
CONFIG="config.json"
RESUME='' #to resume, set as: --resume path/to/checkpoint.pth

#to train
python train_spectrogram_imagenet.py --logdir $LOGDIR"/"$LABEL --config $CONFIG $RESUME

