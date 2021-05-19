import wavencoder
import torch
import numpy as np
import time
import torchvision
from PIL import Image
import pandas as pd
import torchaudio
import sys
import argparse
import math
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa

def fprint(filename, txt):
    with open(filename,"a") as fi:
        print(txt, file=fi)


def segment_nd_features(data, label, segment_size):
    '''
    Segment features into <segment_size> frames.
    Pad with 0 if data frames < segment_size

    Input:
    ------
        - data: shape is (Channel, Time, Freq)
        - label: accent label for the current utterance data
        - segment_size: length of each segment
    
    Return:
    -------
    Tuples of (number of segments, frames, segment labels, utterance label)
        - frames: ndarray of shape (N, C, F, T)
                    - N: number of segments
                    - C: number of input channels
                    - F: frequency index
                    - T: time index
        - segment labels: list of labels for each segments
                    - len(segment labels) == number of segments
    '''
    #  C, T, F
    nch = data.shape[0]
    time = data.shape[1]
    start, end = 0, segment_size
    num_segs = math.ceil(time / segment_size) # number of segments of each utterance
    data_tot = []
    sf = 0
    non_padded_seg_size = []
    for i in range(num_segs):
        # The last segment
        if end > time:
            end = time
            start = max(0, end - segment_size)
        
        non_padded_seg_size.append(end-start)
        
        # Do padding
        data_pad = []
        for c in range(nch):
            data_ch = data[c]
            data_ch = np.pad(
                data_ch[start:end], ((0, segment_size - (end - start)), (0, 0)), mode="constant")
            data_pad.append(data_ch)

        data_pad = np.array(data_pad)
        
        # Stack
        data_tot.append(data_pad)
        
        # Update variables
        start = end
        end = min(time, end + segment_size)
    
    data_tot = np.stack(data_tot)
    utt_label = label
    segment_labels = [label] * num_segs
    
    #Transpose output to N,C,F,T
    data_tot = data_tot.transpose(0,1,3,2)

    return (num_segs, data_tot, non_padded_seg_size)



def main(args):

    #get the paths
    mpath           = args.metapath
    wavpath         = args.trainpath
    testwavpath     = args.testpath
    trainimgpath    = args.trainimgpath
    devimgpath      = args.devimgpath
    testimgpath     = args.testimgpath
    logfile         = 'log.file'
    segmentsize     = 224

    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    #Device
    if torch.cuda.is_available():
        device=torch.device('cuda')
        fprint(logfile, 'use GPU')
    else: 
        device=torch.device('cpu')
        fprint(logfile, 'use CPU')
        
    #wav2vec2
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h").eval().to(device)
    scaler = MinMaxScaler()
    
    #convert train set
    fprint(logfile, "converting train set")
    df = pd.read_csv(mpath+'trainset.csv')
    nutt = 0
    
    for row,col in tqdm(df.iterrows()):
        utt = col['utt']
        fname= col['wavfile']
        label = col['label']
        wav_name = wavpath+fname
    
        #read wav file
        x,sr = librosa.load(wav_name, sr=16000)
            
        #encode
        with torch.no_grad():
            input_values = processor(x, return_tensors="pt").input_values.to(device)
            out = encoder(input_values).last_hidden_state.detach().cpu().numpy()
        
        #flatten and normalize to [0.0, 1.0]
        feat_dim = out.shape[-1]
        wav2vec = out.reshape(-1,1)
        wav2vec = scaler.fit_transform(wav2vec)
        wav2vec = wav2vec.reshape(-1, feat_dim) # T,F
        
        #convert to 8-bit uint
        wav2vec = (wav2vec*255).astype(np.uint8) 
        wav2vec_len = wav2vec.shape[0]


        #save the whole image
        #pil_img = Image.fromarray(wav2vec)
        #pil_img.save(f'{trainimgpath}{label}/{utt}_FULL.png')
        
        #segment
        wav2vec_seg = np.expand_dims(wav2vec, 0) #(1,T,F)
        wav2vec_segmented = segment_nd_features(wav2vec_seg, label, segmentsize)
        num_segments, segments, non_padded_segsize = wav2vec_segmented
        
        #save segments
        if num_segments == 1:   #utterance is <= 4.5 sec
            pil_img = Image.fromarray(np.squeeze(segments[0],0))
            pil_img.save(f'{trainimgpath}{label}/{utt}_single.png')
            nutt += 1
        else:   #utterance is > 4.5sec
            #save the non-overlapping segments        
            for idx, img in enumerate(segments):
                non_padded_size = non_padded_segsize[idx]
            
                if non_padded_size > 50: #only save if the actual segment is > 1sec
                    pil_img = Image.fromarray(np.squeeze(img,0))
                    pil_img.save(f'{trainimgpath}{label}/{utt}_{idx}.png')
                    nutt+= 1
            
            #in addition, take one extra segment from 1.5sec and 1 center crop
            if wav2vec_len > 336:
                pil_img = Image.fromarray(wav2vec[110:110+segmentsize,:].T)
                pil_img.save(f'{trainimgpath}{label}/{utt}_x1.png')
                nutt += 1
                
                pil_img = Image.fromarray(wav2vec.T)
                cropped = torchvision.transforms.functional.center_crop(pil_img, output_size=(feat_dim,segmentsize))
                cropped.save(f'{trainimgpath}{label}/{utt}_cc.png')
                nutt += 1
                
                
    fprint(logfile,f"done - {nutt} segment")


    #convert dev set
    fprint(logfile, "converting dev set")
    df = pd.read_csv(mpath+'devset.csv')
    nutt = 0
    for row,col in tqdm(df.iterrows()):
        utt = col['utt']
        fname= col['wavfile']
        wav_name = wavpath+fname
    
        #read wav file
        x,sr = librosa.load(wav_name, sr=16000)
            
        #encode
        with torch.no_grad():
            input_values = processor(x, return_tensors="pt").input_values.to(device)
            out = encoder(input_values).last_hidden_state.detach().cpu().numpy()
        
        #flatten and normalize to [0.0, 1.0]
        feat_dim = out.shape[-1]
        wav2vec = out.reshape(-1,1)
        wav2vec = scaler.fit_transform(wav2vec)
        wav2vec = wav2vec.reshape(-1, feat_dim).T
        
        #convert to 8-bit uint
        img = Image.fromarray((wav2vec*255).astype(np.uint8), mode='L') 
        img.save(devimgpath+utt+'.png')
        nutt += 1
        

    fprint(logfile,f"done - {nutt} utt")


    #convert test set
    fprint(logfile, "converting test set")
    df = pd.read_csv(mpath+'testset.csv')
    nutt = 0
    for row,col in tqdm(df.iterrows()):
        utt = col['utt']
        fname= col['wavfile']
        wav_name = testwavpath+fname
    
        #read wav file
        x,sr = librosa.load(wav_name, sr=16000)
        
        #encode
        with torch.no_grad():
            input_values = processor(x, return_tensors="pt").input_values.to(device)
            out = encoder(input_values).last_hidden_state.detach().cpu().numpy()
        
        #flatten and normalize to [0.0, 1.0]
        feat_dim = out.shape[-1]
        wav2vec = out.reshape(-1,1)
        wav2vec = scaler.fit_transform(wav2vec)
        wav2vec = wav2vec.reshape(-1, feat_dim).T
        
        #convert to 8-bit uint
        img = Image.fromarray((wav2vec*255).astype(np.uint8), mode='L')
        img.save(testimgpath+utt+'.png')
        nutt += 1
    
    fprint(logfile,f"done - {nutt} utt")
 


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert .wav utterances into spectrogram-based features (default: logspec200)")

    parser.add_argument('--trainpath', type=str, default='audio/',
         help='path to train/dev .wav files')
    parser.add_argument('--testpath', type=str, default='testaudio/',
          help='path to test .wav files')
    parser.add_argument('--metapath', type=str, default='metadata/',
         help='path to metadata files')
    parser.add_argument('--trainimgpath', type=str, default='train_img/',
         help='path to save trainig images')
    parser.add_argument('--devimgpath', type=str, default='dev_img/',
         help='path to save dev images')
    parser.add_argument('--testimgpath', type=str, default='test_img/',
         help='path to save test images')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
