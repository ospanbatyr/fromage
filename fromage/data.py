import torch
import torchvision
import transformers
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop, RandomHorizontalFlip, RandomAffine
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer

from typing import Callable, Optional, Tuple, List, Dict
from PIL import Image, ImageFile, UnidentifiedImageError
import numpy as np
import pydicom as dicom
import os
import os.path as osp
from pathlib import Path
import csv
from .utils import ExpandChannels, load_image

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_ROOT = '/kuacc/users/oince22/hpc_run/physionet.org/files/mimic-cxr-jpg/2.0.0/files/' # TODO needs to change

# values from BioViL repository
RESIZE = 512
CENTER_CROP_SIZE = 480

def image_transform(resize: int, center_crop_size: int, train: bool) -> Compose:
    data_aug_rot = 15
    data_aug_trans = 0.10
    data_aug_scale = 0.10

    if train:
        transforms = [
            Resize(resize), 
            RandomAffine(data_aug_rot, translate=(data_aug_trans, data_aug_trans), scale=(1.0-data_aug_scale, 1.0+data_aug_scale)), 
            CenterCrop(center_crop_size), 
            ToTensor(), 
            ExpandChannels()
        ]
    else:
        transforms = [
            Resize(resize), 
            CenterCrop(center_crop_size), 
            ToTensor(), 
            ExpandChannels()
        ]

    return Compose(transforms)


class MIMICDataset(Dataset):    
    def __init__(self, dataset_path: str, transform: torchvision.transforms):
        self.dataset_path = dataset_path
        self.img_paths, self.reports = self._read_tsv_file()
        self.transform = transform

    def _read_tsv_file(self):
        reports = []
        img_paths = []
        with open(self.dataset_path, "r") as f:
            reader = csv.reader(f, delimiter='\t')
            for report, img_path in reader:
                reports.append(report)
                img_paths.append(Path(osp.join(IMG_ROOT, img_path)))
                
        return img_paths, reports
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        while True:
            try:
                img = load_image(self.img_paths[idx])
                text = self.reports[idx]
                transform_img = self.transform(img)
                return transform_img, text
            except Exception as e:
                print(str(e))
                idx = np.random.randint(0, len(self.img_paths))


class MIMICDataModule(LightningDataModule):
    def __init__(self, config=dict()):
        super().__init__()
        self.config = config
        self._init_img_transform()
        self._init_datasets()
    
    @property
    def loader_config(self):
        default_config = {
            'num_workers': 2,
            'pin_memory': False,
            'batch_size': 64
        }

        return self.config.get('loader', default_config)

    @property
    def dataset_config(self):
        return self.config.get('dataset', dict())

    @property
    def model_config(self):
        return self.config.get('model', dict())

    def _init_img_transform(self) -> None:
        resize = self.config.get('resize', RESIZE)
        center_crop_size = self.config.get('center_crop_size', CENTER_CROP_SIZE)
        self.img_transform = image_transform(resize=resize, center_crop_size=center_crop_size, train=True)

    def _init_datasets(self) -> None:
        dataset_path = self.config.get('dataset_path', 'data/MIMIC_JPG.tsv')
        self.train_data = MIMICDataset(
            dataset_path=dataset_path,
            transform=self.img_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            **self.loader_config
        )
        