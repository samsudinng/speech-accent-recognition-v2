#!/bin/sh
#SBATCH --partition=SCSEGPU_MSAI
#SBATCH --qos=q_msai
#SBATCH --mem=8000M
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --job-name=TestJob
#SBATCH --output=test.out
#SBATCH --error=testError.err

# Set these accordingly
VENV="/home/MSAI/sa0002ng/ACCENT/spectrogram/speech_accent_recognition_new/venv/accent_new_venv"

#1. AlexNetGAP
#LOGDIR=train-logspec-v2
#LABEL=alexnet-196-m5
#CONFIG=config/config_alexnet-196-m5.json
#RESUMEEPOCH=27

#2. VGG16BnGAP
#LOGDIR=train-logspec-v2
#LABEL=vgg16bngap-32-m5
#CONFIG=config/config_vgg16bngap-32-m5.json
#RESUMEEPOCH=5

#3. Resnet18
#LOGDIR=train-logspec-v2
#LABEL=resnet18-32-m5
#CONFIG=config/config_resnet18-32-m5.json
#RESUMEEPOCH=9

#4. Resnet34
#LOGDIR=train-logspec-v2
#LABEL=resnet34-32-m5
#CONFIG=config/config_resnet34-32-m5.json
#RESUMEEPOCH=-1

#5. VGG19BnGAP
#LOGDIR=train-logspec-v2
#LABEL=vgg19bngap-32-m5
#CONFIG=config/config_vgg19bngap-32-m5.json
#RESUMEEPOCH=-1


### WAV2VEC ###
#1. VGG16BnGAP
#LOGDIR=train-wav2vec-v2
#LABEL=vgg16bngap-16-m5
#CONFIG=config/config_wav2vec_vgg16bngap-16-m5.json
#RESUMEEPOCH=3

#2. AlexNetGAP
#LOGDIR=train-wav2vec-v2
#LABEL=alexnet-16-m5
#CONFIG=config/config_wav2vec_alexnet-16-m5.json
#RESUMEEPOCH=18

#3. Resnet18
#LOGDIR=train-wav2vec-v2
#LABEL=resnet18-16-m5
#CONFIG=config/config_wav2vec_resnet18-16-m5.json
#RESUMEEPOCH=-1

#4. Resnet50
#LOGDIR=train-wav2vec-v2
#LABEL=resnet50-16-m5
#CONFIG=config/config_wav2vec_resnet50-16-m5.json
#RESUMEEPOCH=-1

#5. VGG19BnGAP
#LOGDIR=train-wav2vec-v2
#LABEL=vgg19bngap-16-m5
#CONFIG=config/config_wav2vec_vgg19bngap-16-m5.json
#RESUMEEPOCH=0

#6. MobilenetV2
#LOGDIR=train-wav2vec-v2
#LABEL=mobilenetv2-16-m5
#CONFIG=config/config_wav2vec_mobilenetv2-16-m5.json
#RESUMEEPOCH=-1


### WAV2VEC2 ###
#1. AlexNetGAP
#LOGDIR=train-wav2vec2
#LABEL=alexnet-16-m5
#CONFIG=config/config_wav2vec2_alexnet-16-m5.json
#RESUMEEPOCH=82
#LOGDIR=train-wav2vec2-xlsr
#LABEL=alexnet-16-m4-nonorm
#CONFIG=config/wav2vec2-xlsr-nonorm/config_wav2vec2xlsr_alexnet-16-m4.json
#RESUMEEPOCH=-1

#2. VGG16BnGAP
#LOGDIR=train-wav2vec2
#LABEL=vgg16bngap-16-m5
#CONFIG=config/config_wav2vec2_vgg16bngap-16-m5.json
#RESUMEEPOCH=10
LOGDIR=train-wav2vec2-xlsr-3ch
LABEL=vgg16bngap-16-m4
CONFIG=config/wav2vec2-xlsr/config_wav2vec2xlsr-3ch_vgg16bn-16-m4.json
RESUMEEPOCH=8

#3. Resnet50
#LOGDIR=train-wav2vec2
#LABEL=resnet50-16-m5
#CONFIG=config/config_wav2vec2_resnet50-16-m5.json
#RESUMEEPOCH=-1


# activate the venv
source $VENV/bin/activate

# execute
if [ $RESUMEEPOCH == -1 ] 
then
    echo "Train"
    python train_accent_imagenet.py --logdir $LOGDIR"/"$LABEL \
                                         --config $CONFIG

else
    CHECKPOINT="checkpoint_spectrogram_epoch"$RESUMEEPOCH".pth"

    #get path of checkpoint file
    if [[ -f $LOGDIR/$LABEL/$CHECKPOINT ]] 
    then
        CHECKPOINTPATH=$LOGDIR/$LABEL/$CHECKPOINT
        echo "Checkpoint: "$CHECKPOINTPATH
    else
        #search in previously resumed directories
        CHECKPOINTPATH=$(find $LOGDIR/$LABEL-r* -type f -name $CHECKPOINT)
        if [[ ! -f $CHECKPOINTPATH ]] 
        then
            echo "Checkpoint not found"
            CHECKPOINTPATH="error!"
        fi
    fi
    
    echo "Resume from checkpoint: "$CHECKPOINTPATH
    python train_accent_imagenet.py --logdir $LOGDIR"/"$LABEL"-r"$RESUMEEPOCH \
                                         --config $CONFIG \
                                         --resume $CHECKPOINTPATH

fi

