import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import json
from PIL import Image
import json
from pathlib import Path
try:
    from .utils import load_image
except:
    from utils import load_image
    
class VQADataset(data.Dataset):
    def __init__(self, data_path, image_transform=None):
        self.data_path = data_path
        self.image_transform = image_transform # can be implemented later

    def __len__(self):
        return self.get_len()

    def __getitem__(self, index):
        
        image = self.get_image(index, self.image_transform)
        if self.image_transform != None:
            image = self.image_transform(image)
        question = self.get_question(index)
        answer = self.get_answer(index)

        return image, question, answer
    
    
class VQA_RADDataset(VQADataset):
    def __init__(self, data_path, image_transform=None, q_filter=None):
        super(VQADataset, self).__init__()
        
        self.data_path = data_path
        self.image_transform = image_transform
        self.data = self.load_data('/VQA_RAD Dataset Public.json', q_filter)
        
    def load_data(self, filename, q_filter):
        f = open(self.data_path + filename)
        data = json.load(f)
        
        # filter the data to only retain chest images
        data = [d for d in data if d.get('image_organ') == 'CHEST']
        
        if q_filter == 'closed':
            data = [d for d in data if d.get('answer_type') == 'CLOSED']
        if q_filter == 'open':
            data = [d for d in data if d.get('answer_type') == 'OPEN']
        
        return data
    
    def get_len(self):
        return len(self.data)
    
    def get_image(self, index, image_transform=None):
        image_path = Path(self.data_path + '/VQA_RAD Image Folder/' + self.data[index].get('image_name'))
        return load_image(image_path)
    
    def get_question(self, index):
        return self.data[index].get('question')
    
    def get_answer(self, index):
        return self.data[index].get('answer')
