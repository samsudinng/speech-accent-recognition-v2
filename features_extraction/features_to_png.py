import numpy as np
import pandas as pd
import pickle, gzip
from tqdm import tqdm
from PIL import Image
import math
import gc
import sys
import argparse
import os

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
                #data_ch[start:end], ((0, segment_size - (end - start)), (0, 0)), mode="constant",
                #constant_values=((-80,-80),(-80,-80)))
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



def convert_to_train_image_v1(allspec, data_labels, img_dir,segmentsize=300):
    """
    Perform the following:
        1. convert values to 8-bit pixel, segment 
        2. save as .png image
    """
    for utt,logspec in tqdm(allspec.items()):
        #get the utterance label
        label = data_labels.loc[utt,'label']        

        #segment
        logspec = np.expand_dims(logspec, 0).transpose(0,2,1) #(1,T,F)
        logspec_segmented = segment_nd_features(logspec, label, segmentsize)

        num_segments, segments, _ = logspec_segmented

        for idx, img in enumerate(segments):
            pil_img = Image.fromarray(np.squeeze(img,0))
            pil_img.save(f'{img_dir}{label}/{utt}_{idx}.png')

        data_labels.loc[utt,'num_segments'] = num_segments

    return data_labels



def convert_to_train_image_v2(allspec, data_labels, img_dir,segmentsize=300):
    """
    Perform the following:
        1. convert values to 8-bit pixel, segment 
        2. save as .png image
    """
    for utt,logspec in tqdm(allspec.items()):
        total_seg = 0
        
        #get the utterance label
        label = data_labels.loc[utt,'label']        
        
        #segment
        logspec_seg = np.expand_dims(logspec, 0).transpose(0,2,1) #(1,T,F)
        logspec_segmented = segment_nd_features(logspec_seg, label, segmentsize)
        num_segments, segments, non_padded_segsize = logspec_segmented
        
        if num_segments == 1:   #utterance is <= 3 sec
            pil_img = Image.fromarray(np.squeeze(segments[0],0))
            pil_img.save(f'{img_dir}{label}/{utt}_single.png')
            total_seg += 1
        else:   #utterance is > 3sec
            #save the non-overlapping segments        
            for idx, img in enumerate(segments):
                non_padded_size = non_padded_segsize[idx]
            
                if non_padded_size > 100: #only save if the actual segment is > 1sec
                    pil_img = Image.fromarray(np.squeeze(img,0))
                    pil_img.save(f'{img_dir}{label}/{utt}_{idx}.png')
                    total_seg += 1
            
            #in addition, save the whole utterance
            pil_img = Image.fromarray(logspec)
            pil_img.save(f'{img_dir}{label}/{utt}.png')
            total_seg += 1
        
        data_labels.loc[utt,'num_segments'] = total_seg

    return data_labels


def convert_to_image(allspec, data_labels, img_dir):
    """
    Perform the following:
        1. convert values to 8-bit pixel, segment 
        2. save as .png image
    """
    for utt,logspec in tqdm(allspec.items()):
        
        #save the whole utterance
        pil_img = Image.fromarray(logspec)
        pil_img.save(f'{img_dir}{utt}.png')
        
        
        data_labels.loc[utt,'num_segments'] = 1

    return data_labels



def main(args):

    feature      = args.feature
    to_delete    = args.delete
    xtract_train = args.xtrain
    xtract_test  = args.xtest
    
    #folder containing .pkl.gz spectrogram images
    train_feature_dir  = args.fpath
    dev_feature_dir    = args.fpath
    test_feature_dir   = args.fpath
    
    #utt2accent
    train_label_file = args.metapath+'train_utt2accent'
    dev_label_file   = args.metapath+'dev_utt2accent'
    test_label_file   = args.metapath+'test_utt2accent'

    #number of .pkl.gz file segments in the train set
    n_train_feats_files = args.trainsplit
    n_dev_feats_files   = args.devsplit
    n_test_feats_files   = args.testsplit
    
    #directory to save the resulting train images
    train_dir = args.trainpath
    dev_dir   = args.devpath
    test_dir   = args.testpath
    
    segmentsize = args.segmentsize
    
    #Read the label data into dataframe
    train_labels = pd.read_csv(train_label_file, header=None, delim_whitespace=True,index_col=0)
    train_labels.columns = ['label']
    train_labels['num_segments']=np.nan
    dev_labels = pd.read_csv(dev_label_file, header=None, delim_whitespace=True,index_col=0)
    dev_labels.columns = ['label']
    dev_labels['num_segments']=np.nan
    test_labels = pd.read_csv(test_label_file, header=None, delim_whitespace=True,index_col=0)
    test_labels.columns = ['label']
    test_labels['num_segments']=np.nan
    print(f'Train set utt - {len(train_labels)}')
    print(f'Dev   set utt - {len(dev_labels)}')
    print(f'Test  set utt - {len(test_labels)}')

    #Read the training features, segment and put in train_img folder with torchvision ImageFolder dataset format.
    #Also save the image of the whole utterance as augmentation.
    #   Format of .pkl.gz content: {utterance:log_spectrogram}
    #       - utterance: utterance name as in utt2accent, eg.AESRC2020-AMERICAN-ACCENT-G00473-G00473S1001
    #       - log_spectrogram: (F, T), standardized and quantized to uint8
    if xtract_train == 1:
        print("Converting train features to png")
        for nf in range(n_train_feats_files):
            filename = f'{train_feature_dir}train_{feature}_{nf}.pkl.gz'
            print(f'Converting {filename} to .png ...')
            with gzip.open(filename, 'rb') as f:
                allspec = pickle.load(f)
            
            if args.version == 'v1':
                train_labels = convert_to_train_image_v1(allspec, train_labels, train_dir,segmentsize=segmentsize)
            elif args.version == 'v2':
                train_labels = convert_to_train_image_v2(allspec, train_labels, train_dir,segmentsize=segmentsize)        
            
            del allspec
            gc.collect()
            if to_delete == 1:
                os.remove(filename)

        dropped  = len(train_labels) 
        train_labels.dropna(inplace=True)
        dropped -= len(train_labels)
        assert train_labels.num_segments.isna().any().item() == False
        train_labels.to_pickle(f'{train_dir}train_labels.pkl')
        print(f'Train set - num. utt: {len(train_labels)} - {dropped} utterances dropped')
    
    #Read the dev features and save as .png in dev_img folder
    if xtract_train == 1:
        print("Converting dev features to png")
        for nf in range(n_dev_feats_files):
            filename = f'{dev_feature_dir}dev_{feature}_{nf}.pkl.gz'
            print(f'Converting {filename} to png ...')
            with gzip.open(filename, 'rb') as f:
                allspec = pickle.load(f)
            dev_labels = convert_to_image(allspec, dev_labels, dev_dir) 

            del allspec
            gc.collect()
            if to_delete == 1:
                os.remove(filename)
        
        dropped  = len(dev_labels) 
        dev_labels.dropna(inplace=True)
        dropped -= len(dev_labels)
        assert dev_labels.num_segments.isna().any().item() == False
        dev_labels.to_pickle(f'{dev_dir}dev_labels.pkl')
        print(f'Dev   set - num. utt: {len(dev_labels)} - {dropped} utterances dropped')

    #Read the test features and save as .png in test_img folder
    if xtract_test == 1:
        print("Converting test features to png")
        for nf in range(n_test_feats_files):
            filename = f'{test_feature_dir}test_logspec200_{nf}.pkl.gz'
            print(f'Converting {filename} to png ...')
            with gzip.open(filename, 'rb') as f:
                allspec = pickle.load(f)
            test_labels = convert_to_image(allspec, test_labels, test_dir) 

            del allspec
            gc.collect()
            if to_delete == 1:
                os.remove(filename)
                    
        dropped  = len(test_labels) 
        test_labels.dropna(inplace=True)
        dropped -= len(test_labels)
        assert test_labels.num_segments.isna().any().item() == False
        test_labels.to_pickle(f'{test_dir}test_labels.pkl')
        print(f'Test  set - num. utt: {len(test_labels)} - {dropped} utterances dropped')

    


def parse_arguments(argv):

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Segment and convert spectrogram-based features into .png image files")
    parser.add_argument('--feature', type=str, default='logspec200',
        help='Features to be extracted')
    parser.add_argument('--trainpath', type=str, default='train_img/',
         help='path to train set output images')
    parser.add_argument('--version', type=str, default='v1',
         help='version of training images')
    parser.add_argument('--devpath', type=str, default='dev_img/',
          help='path to dev set output images')
    parser.add_argument('--testpath', type=str, default='test_img/',
          help='path to test set output images')
    parser.add_argument('--fpath', type=str, default='features/',
         help='path to feature files')
    parser.add_argument('--metapath', type=str, default='metadata/',
         help='path to metadata files')
    parser.add_argument('--segmentsize', type=int, default=300,
         help='segment size')
    parser.add_argument('--trainsplit', type=int, default=8,
         help='number of train set split')
    parser.add_argument('--devsplit', type=int, default=1,
         help='number of dev set split')
    parser.add_argument('--testsplit', type=int, default=1,
         help='number of test set split')
    parser.add_argument('--xtrain', type=int, default=1,
         help='set 1 to extract train\dev')
    parser.add_argument('--xtest', type=int, default=1,
         help='set 1 to extract test')
    parser.add_argument('--delete', type=int, default=0,
             help='set to 1 to delete features files after conversion')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
