import os
import random
import torch
import numpy as np
import PIL.Image as Image



from torch.utils.data import Dataset
from torchvision import transforms, utils

from torch.nn.utils.rnn import pad_sequence

class loadedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))
        self.count = [len(os.listdir(self.root_dir + '/' + c)) for c in self.classes]
        self.acc_count = [self.count[0]]
        for i in range(1, len(self.count)):
                self.acc_count.append(self.acc_count[i-1] + self.count[i])
    def __len__(self):
        l = np.sum(np.array([len(os.listdir(self.root_dir + '/' + c)) for c in self.classes]))
        return l
    
    def get_labels(self):
        return self.classes
    
    def __getitem__(self, idx):
        label = 1
        for i in range(len(self.acc_count)):
            if idx < self.acc_count[i]:
                label = i
                break
        class_path = self.root_dir + '/' + self.classes[label] 
        if label:
            file_path = class_path + '/' + sorted(os.listdir(class_path))[idx-self.acc_count[label]]
        else:
            file_path = class_path + '/' + sorted(os.listdir(class_path))[idx]

        _, file_name = os.path.split(file_path)

        frames = []
        file_list = sorted(os.listdir(file_path))
        for i, f in enumerate(file_list):
            npyfile = os.path.join(file_path + '/' + f)          
            frame = torch.from_numpy(np.load(npyfile))            
            if self.transform is not None:
                frame = self.transform[0](frame)
            frames.append(frame)
            if len(frames) == 8:
                break
        while len(frames) < 8:
            frames.append(frame)
        return frames, label, file_name
