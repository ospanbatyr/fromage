import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import json
from PIL import Image
import json
import pandas as pd
from pathlib import Path
from .utils import load_image
import os.path as osp
    
class ImgClsDataset(data.Dataset):
    def __init__(self, data_path, image_transform=None):
        self.data_path = data_path
        self.image_transform = image_transform # can be implemented later

    def __len__(self):
        return self.get_len()

    def __getitem__(self, index):
        image = self.get_image(index, self.image_transform)
        if self.image_transform != None:
            image = self.image_transform(image)
            
        img_class = self.get_class(index)
        return image, img_class
    
    
class RSNAPneumoniaDataset(ImgClsDataset):
    def __init__(self, data_path, image_transform=None):
        super(ImgClsDataset, self).__init__()
        
        self.data_path = data_path
        self.image_transform = image_transform
        self.data = pd.read_csv(osp.join(self.data_path, "stage_2_train_labels_short.csv"))

    def get_len(self):
        return len(self.data)
    
    def get_image(self, index, image_transform=None):
        image_path = Path(osp.join(self.data_path, 'stage_2_train_images', self.data['patientId'].iloc[index]) + '.dcm')
        return load_image(image_path)
    
    def get_class(self, index):
        return self.data['Target'].iloc[index]

    def __getitem__(self, index):
        image = self.get_image(index, self.image_transform)
        if self.image_transform != None:
            image = self.image_transform(image)
            
        img_class = "yes" if self.get_class(index) == 1 else "no"
        return image, img_class
    

class COVIDDataset(ImgClsDataset):
    def __init__(self, data_path, image_transform=None):
        super(ImgClsDataset, self).__init__()
        
        self.data_path = data_path
        self.image_transform = image_transform
        
        with open(osp.join(self.data_path, "COVIDshort.json")) as f:
            self.data = json.load(f)
    
    def get_len(self):
        return len(self.data)
    
    def get_image(self, index, image_transform=None):
        image_path = Path(osp.join(self.data_path, self.data[index].get('image_path')))
        return load_image(image_path)
    
    def get_class(self, index):
        return self.data[index].get('class')

    def __getitem__(self, index):
        image = self.get_image(index, self.image_transform)
        if self.image_transform != None:
            image = self.image_transform(image)
            
        img_class = self.get_class(index)
        img_class = "Pneumonia" if img_class == "Non-COVID" else img_class
        return image, img_class
