import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import json
from PIL import Image
import json
import pandas as pd
from pathlib import Path
try:
    from .utils import load_image
except:
    from utils import load_image
    
class imgclsDataset(data.Dataset):
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
    
    
class RSNAPneumoniaDataset(imgclsDataset):
    def __init__(self, data_path, image_transform=None):
        super(imgclsDataset, self).__init__()
        
        self.data_path = data_path
        self.image_transform = image_transform
        self.data = self.load_data('/stage_2_train_labels_short.csv')
        
    def load_data(self, filename):
        data = pd.read_csv(self.data_path + filename)
        # shorten the csv
        return data
    
    def get_len(self):
        return len(self.data)
    
    def get_image(self, index, image_transform=None):
#         print(self.data['patientId'])
#         print('index: ', index)
        
        image_path = Path(self.data_path + '/stage_2_train_images/' + self.data['patientId'].iloc[index] + '.dcm')
        return load_image(image_path)
    
    def get_class(self, index):
        return self.data['Target'].iloc[index]
    

class COVIDDataset(imgclsDataset):
    def __init__(self, data_path, image_transform=None):
        super(imgclsDataset, self).__init__()
        
        self.data_path = data_path
        self.image_transform = image_transform
        self.data = self.load_data('/COVIDshort.json')
        
    def load_data(self, filename):
        f = open(self.data_path + filename)
        data = json.load(f)
        return data
    
    def get_len(self):
        return len(self.data)
    
    def get_image(self, index, image_transform=None):
        image_path = Path(self.data_path + "/" + self.data[index].get('image_path'))
        return load_image(image_path)
    
    def get_class(self, index):
        return self.data[index].get('class')
