import os
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from preprocess.audio_process import *
from preprocess.image_process import *

np.random.seed(42)

class UAVLoader(Dataset):
    def __init__(self, annotation_lines,root_path,dark_aug=0,testing=0):  
        super(UAVLoader, self).__init__()
        self.annotation_lines   = annotation_lines
        self.audio_path         = os.path.join(root_path,'audio_npy')
        self.gt_path            = os.path.join(root_path,'gt')
        self.pseudo_label_path       = os.path.join(root_path,'pseudo_label')

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):

        name = self.annotation_lines[index].strip()
        # Robustly extract the base path without extension
        base_name, _ = os.path.splitext(name)
        
        audio_name  = os.path.join(self.audio_path, base_name + '.npy')
        gt_name     = os.path.join(self.gt_path, base_name + '.npy')
        pseudo_label_name = os.path.join(self.pseudo_label_path, base_name + '.npy')
        
        audio   = make_seq_audio(self.audio_path, base_name + '.npy')
        audio   = np.transpose(audio,[1,0])
        spec       = Audio2Spectrogram(audio,sr=46080)
        spec       = spec.float()

        gt = np.load(gt_name)
        pseudo_label = np.load(pseudo_label_name)

        # Standard Scale Target Normalization for high-stability regression convergence
        mean = np.array([2.370453, 7.071116, 9.55415], dtype=np.float32)
        std = np.array([2.0044913, 5.648387, 5.4611444], dtype=np.float32)

        gt = (gt - mean) / std
        pseudo_label = (pseudo_label - mean) / std

        gt = torch.from_numpy(gt).float()
        pseudo_label = torch.from_numpy(pseudo_label).float()

        return spec, gt, pseudo_label

