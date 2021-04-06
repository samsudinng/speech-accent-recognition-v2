import torch
import torchaudio
import torchvision
import pandas as pd
import os
from PIL import Image
import numpy as np


class AccentImageTESTDataset(torch.utils.data.Dataset):

    """
    AESRC2020 speech accent dev/test dataset.    
        - read image
        - zero-pad the image to 3sec if shorter
        - apply transform
    """

    
    def __init__(self, csv_file, root_dir,min_tlen=300, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.utterances = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.num_utt = len(self.utterances)
        self.min_tlen = min_tlen

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        
        fname = os.path.join(self.root_dir, self.utterances.utt.iloc[idx]+'.png')
        x = Image.open(fname).convert(mode='RGB')

        #pad to min length for smaller segment
        padlen = self.min_tlen - x.size[0]
        if padlen > 0:
            x = torchvision.transforms.functional.pad(x,[0,0,padlen,0])
        
        if self.transform:
            x = self.transform(x)

        label = self.utterances.label.iloc[idx]
        return x, label






