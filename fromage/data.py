import torch
import torchvision
import transformers
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop, RandomHorizontalFlip, RandomAffine, Normalize, RandomCrop
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
import json
from .utils import ExpandChannels, load_image
from random import choices
from copy import deepcopy

ImageFile.LOAD_TRUNCATED_IMAGES = True

# values from BioViL repository
RESIZE = 512
CENTER_CROP_SIZE = 480

def cxr_image_transform(resize: int, center_crop_size: int, train: bool) -> Compose:
    data_aug_rot = 15
    data_aug_trans = 0.10
    data_aug_scale = 0.10

    if train:
        transforms = [
            RandomAffine(data_aug_rot, translate=(data_aug_trans, data_aug_trans), scale=(1.0-data_aug_scale, 1.0+data_aug_scale)),
            Resize(resize), 
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


def coco_image_transform(train: bool):
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        img_transform = Compose([
            Resize(232),
            RandomCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ])
    else:
        img_transform = Compose([
            Resize(232),
            CenterCrop(224),
            ToTensor(),
            normalize,
        ])
    return img_transform


class MIMICDataset(Dataset):    
    def __init__(self, dataset_path: str, img_path: str, transform: torchvision.transforms):
        self.dataset_path = dataset_path
        self.img_path = img_path
        self.data, self.length = self._read_tsv_file()
        self.transform = transform

    def _preprocess_img_view(self, txt):
        paths, positions = [], []
        study_views = txt.split("[VIEW_DELIM]")
        for view in study_views:
            path, pos = view.split("[LOC_DELIM]")
            paths.append(Path(osp.join(self.img_path, path)))
            positions.append(pos)
        
        return paths, positions

    def _preprocess_img_text(self, txt):
        if "[NEXT_IMG]" in txt:
            cur_study_txt, next_study_txt = txt.split("[NEXT_IMG]")
            next_paths, next_pos = self._preprocess_img_view(next_study_txt)
        else:
            cur_study_txt = txt
            next_paths, next_pos = "", ""

        cur_paths, cur_pos = self._preprocess_img_view(cur_study_txt) # shuffle or not, thats the question
        return cur_paths, cur_pos, next_paths, next_pos

    def _read_tsv_file(self):
        data = {}
        with open(self.dataset_path, "r") as f:
            reader = csv.reader(f, delimiter='\t')
            for report, img_path, cur_img_count, next_img_count in reader:
                if "[NEXT_TXT]" in report:
                    cur_report, next_report = report.split("[NEXT_TXT]")
                else:
                    cur_report, next_report = report, ""
                
                cur_paths, cur_pos, next_paths, next_pos = self._preprocess_img_text(img_path)
                cur_item = {
                    "cur_report": cur_report,
                    "next_report": next_report,
                    "cur_paths": cur_paths,
                    "cur_pos": cur_pos,
                    "next_paths": next_paths,
                    "next_pos": next_pos
                }

                if f"{cur_img_count}|{next_img_count}" not in data:
                    data[f"{cur_img_count}|{next_img_count}"] = [cur_item]
                else:
                    data[f"{cur_img_count}|{next_img_count}"].append(cur_item)
        
        length = 0
        for img_views, items in data.items():
            length += len(items)

        return data, length
        
    def __len__(self):
        return self.length

    def lens(self):
        return {img_views: len(items) for img_views, items in self.data.items()}

    def __getitem__(self, item):
        idx, cur_img, next_img = item["idx"], item["cur_img"], item["next_img"]
        img_view = f"{cur_img}|{next_img}"
        item = self.data[img_view][idx]
        cur_report = item["cur_report"]
        next_report = item["next_report"]
        cur_imgs = torch.stack([self.transform(load_image(cur_path)) for cur_path in item["cur_paths"]])
        if item["next_paths"]:
            next_imgs = torch.stack([self.transform(load_image(next_path)) for next_path in item["next_paths"]])
        else:
            next_imgs = torch.empty((0, ))
        return cur_imgs, next_imgs, cur_report, next_report


class COCODataset(Dataset):
    def __init__(self, path, split, year, image_transform):
        super().__init__()
        assert split in ('train', 'val')
        assert year in (2014, 2017)

        self.year = year
        self.split = split
        self.image_transform = image_transform
        self.path = osp.abspath(osp.expanduser(path))
        
        self.setup_dataset()

    def setup_dataset(self):
        self.split_name = f'{self.split}{self.year}'
        self.image_dir = osp.join(self.path, self.split_name)
        self.annotation_file = osp.join(self.path, 'annotations', f'captions_{self.split_name}.json')

        with open(self.annotation_file, "r") as f:
            json_data = json.load(f)
            annotations = json_data['annotations']

        image_dict = dict()
        for item in json_data['images']:
            image_dict[item['id']] = item

        self.annotations = annotations
        self.image_dict = image_dict

    def __len__(self):
        return len(self.annotations)

    def _read_image(self, idx):
        image_id = self.annotations[idx]['image_id']
        file_name = self.image_dict[image_id]['file_name']
        file_path = osp.join(self.image_dir, file_name)
        raw = Image.open(file_path)
        raw = raw.convert("RGB") if raw.mode != "RGB" else raw
        
        image = self.image_transform(raw)
        return image

    def __getitem__(self, idx):
        while True:
            try:
                image = self._read_image(idx)
                caption = self.annotations[idx]['caption']
                return image, caption
            except Exception as e:
                print(str(e))
                idx = np.random.randint(0, len(self))


class BaseDataModule(LightningDataModule):
    def __init__(self, config=dict()):
        super().__init__()
        self.config = config
        self._init_img_transform()
        self._init_datasets()
    
    @property
    def loader_config(self):
        return self.config['loader']

    @property
    def dataset_config(self):
        return self.config['dataset']

    @property
    def model_config(self):
        return self.config['model']

    def _init_img_transform(self):
        raise NotImplementedError("Implement '_init_img_transform'")

    def _init_datasets(self):
        raise NotImplementedError("Implement '_init_datasets'")


class CaptionDataModule(BaseDataModule):
    def __init__(self, config=dict()):
        super(CaptionDataModule, self).__init__(config)

    def _init_img_transform(self) -> None:
        self.train_img_transform = coco_image_transform(train=True)
        self.val_img_transform = coco_image_transform(train=False)

    def _init_datasets(self) -> None:
        dataset_path = self.dataset_config['path']
        year = self.dataset_config['year']
        self.train_data = COCODataset(dataset_path, "train", year, self.train_img_transform)
        self.val_data = COCODataset(dataset_path, "val", year, self.val_img_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            **self.loader_config
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            shuffle=False,
            **self.loader_config
        )

    def predict_dataloader(self):
        return self.val_dataloader()


class GroupSampler(Sampler):
    def __init__(self, data, lens, batch_size, shuffle=True):
        self.data = data
        self.lens = lens
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_views, self.probs, self.batch_counts = self._sample_prob()
        self.batches = self._create_batches()

    def _sample_prob(self):
        img_views = []
        probs = []
        batch_counts = 0

        length = sum(view_len for view_len in self.lens.values())
        for img_view, view_len in self.lens.items():
            img_views.append(img_view)
            probs.append(view_len / length)
            batch_counts += view_len // self.batch_size

        return img_views, probs, batch_counts

    def _create_batches(self):
        indices = []
        states = [0 for _ in range(len(self.img_views))]
        batches = [] # tuple, split and batch
        probs = [fp for fp in self.probs] 

        for img_view in self.img_views:
            if self.shuffle:
                indices.append(torch.randperm(len(self.data[img_view])))
            else:
                indices.append(torch.arange(len(self.data[img_view])))

        splits = list(range(len(self.img_views)))

        while True:
            if not splits:
                break
            
            cur_img_view = choices(splits, weights=probs)[0]
            if (states[cur_img_view] + self.batch_size) > self.lens[self.img_views[cur_img_view]]:
                idx = splits.index(cur_img_view)
                del splits[idx]
                del probs[idx]
                continue

            cur_indices = indices[cur_img_view][states[cur_img_view]:states[cur_img_view] + self.batch_size]
            cur_imgs = [int(self.img_views[cur_img_view].split("|")[0]) for _ in range(self.batch_size)]
            next_imgs = [int(self.img_views[cur_img_view].split("|")[1]) for _ in range(self.batch_size)]
            batch = [{"idx": idx, "cur_img": cur_img, "next_img": next_img} for idx, cur_img, next_img in zip(cur_indices, cur_imgs, next_imgs)]

            batches.append(batch)
            states[cur_img_view] += self.batch_size

        return batches
            
    def __iter__(self):
        for i in range(0, self.batch_counts):
            yield self.batches[i]

    def __len__(self):
        return len(self.batch_counts)


class MIMICDataModule(BaseDataModule):
    def __init__(self, config=dict()):
        super(MIMICDataModule, self).__init__(config)
    
    def _init_img_transform(self) -> None:
        resize = self.dataset_config['resize']
        center_crop_size = self.dataset_config['center_crop_size']
        self.train_img_transform = cxr_image_transform(resize=resize, center_crop_size=center_crop_size, train=True)
        self.val_img_transform = cxr_image_transform(resize=resize, center_crop_size=center_crop_size, train=False)

    def _init_datasets(self) -> None:
        tsv_path = self.dataset_config['tsv_path']
        img_path = self.dataset_config['img_path']

        train_tsv_path = tsv_path.replace("<SPLIT>", "train")
        valid_tsv_path = tsv_path.replace("<SPLIT>", "valid")
        test_tsv_path = tsv_path.replace("<SPLIT>", "test")

        self.train_data = MIMICDataset(
            dataset_path=train_tsv_path,
            img_path=img_path,
            transform=self.train_img_transform
        )
        self.val_data = MIMICDataset(
            dataset_path=valid_tsv_path,
            img_path=img_path,
            transform=self.val_img_transform
        )
        self.test_data = MIMICDataset(
            dataset_path=test_tsv_path,
            img_path=img_path,
            transform=self.val_img_transform
        )

    def train_dataloader(self):
        batch_sampler = GroupSampler(self.train_data.data, self.train_data.lens(), self.loader_config["batch_size"], shuffle=True)
        config = self.loader_config
        del config["batch_size"]

        return DataLoader(
            self.train_data,
            batch_sampler=batch_sampler,
            **config
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            shuffle=False,
            **self.loader_config
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            shuffle=False,
            **self.loader_config
        )

    def predict_dataloader(self):
        return self.val_dataloader()

        