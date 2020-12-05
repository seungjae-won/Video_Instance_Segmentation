import os
import numpy as np
import torch
import torch.nn as nn

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, data_dir, image_type, transform=None):
        self.transform = transform
        #self.input_list = sorted(os.listdir(os.path.join(self.data_dir,'train')))
        #self.label_list = sorted(os.listdir(os.path.join(self.data_dir,'label')))
        
        if mode == 'train':
            if image_type == 'gray':
                self.input_list = np.load(data_dir+'/train_gray.npy')
            elif image_type == 'rgb':
                self.input_list = np.load(data_dir+'/train_rgb.npy')
            self.label_list = np.load(data_dir+'/label.npy')
        else:
            if image_type == 'gray':
                self.input_list = np.load(data_dir+'/test_gray.npy')
            elif image_type == 'rgb':
                self.input_list = np.load(data_dir+'/test_rgb.npy')
                
    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, index):
        input = self.input_list[index]
        label = self.label_list[index]
        
        input = input/255.0
        label = label/255.0
        
        data = {'input' : input, 'label' : label}
        
        if self.transform:
            data = self.transform(data)
        return data
    
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

    
    
    
    
    
    
    