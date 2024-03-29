import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Callable, Optional, Tuple, List, Dict
import numpy as np
import pydicom as dicom
from pathlib import Path
from skimage import io
import re
import os
import string
import warnings
import os.path as osp
from PIL import Image, ImageFile, UnidentifiedImageError
from shutil import copy
from omegaconf import OmegaConf
from collections import OrderedDict


ImageFile.LOAD_TRUNCATED_IMAGES = True


class RETTokenCallback(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        ret_token_idx = pl_module.model.ret_token_idx

        for param in pl_module.model.model.input_embeddings.parameters():
            mask = torch.arange(param.grad.shape[0]) != ret_token_idx
            param.grad[mask,:] = 0.0


def save_model_path(config):
    return osp.join(config['logger']['save_dir'], 'checkpoints', config['logger']['name'], "last.ckpt")

class SaveModelCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        config = pl_module.config
        param_dict = OrderedDict()
        for name, param in pl_module.model.model.state_dict().items():
            if not (name.startswith("lm") or name.startswith("vm")):
                param_dict[name] = param

        torch.save(param_dict, save_model_path(config))


def save_config(config, log_dir):
    checkpoints_path = osp.join(log_dir, 'checkpoints', config['logger']['name'])
    os.makedirs(checkpoints_path, exist_ok=True)
    
    config_name = "config.yaml"
    config_path = osp.join(checkpoints_path, config_name)
    
    OmegaConf.save(config=dict(config), f=config_path)
    print(f"Saved the config to {config_path}")


def create_callbacks(config, log_dir):
    checkpoints_path = osp.join(log_dir, 'checkpoints', config['logger']['name'])
    os.makedirs(checkpoints_path, exist_ok=True)

    return [RETTokenCallback(), SaveModelCallback()]


def create_logger(config):
    assert config['logger'].get('version') is not None, "Logger version should not be None"
    if config['logger']['version'] == 'debug':
        return None
    config['logger']['save_dir'] = osp.abspath(
        osp.expanduser(config['logger']['save_dir']))
    if config['logger']['project'] is None:
        architecture = "fromage"
        config['logger']['project'] = f'{architecture}'
    logger = pl.loggers.WandbLogger(**config['logger'])
    return logger


def contrastive_loss(logits: torch.Tensor):
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def get_logits(t2i_embs: torch.Tensor, i2t_embs: torch.Tensor):
    logits_per_image = i2t_embs @ t2i_embs.t()
    logits_per_text = logits_per_image.t()
    return logits_per_image, logits_per_text


def retrieval_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor):
    caption_loss = contrastive_loss(logits_per_text)
    image_loss = contrastive_loss(logits_per_image)
    return (caption_loss + image_loss) / 2.0


def accuracy(output: torch.Tensor, target: torch.Tensor, padding: int, topk: tuple):
    with torch.no_grad():
        maxk = max(topk)
        if output.shape[-1] < maxk:
            warnings.warn("Output example count is smaller than maxk", Warning)
        
        maxk = min(maxk, output.shape[-1])
        bsz = target.shape[0]

        # Take topk along the last dimension
        _, pred = output.topk(maxk, -1, largest=True, sorted=True)
        
        mask = (target != padding).type(target.dtype)
        target_expand = target[..., None].expand_as(pred)
        correct = pred.eq(target_expand)
        correct = correct * mask[..., None].expand_as(correct)

        res = []
        for k in topk:
            correct_k = correct[..., :k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / mask.sum()))

        return res


def contrastive_acc(logits: torch.Tensor, topk: tuple) -> torch.Tensor:
    target = torch.arange(len(logits), device=logits.device)
    return accuracy(logits, target, -1, topk)


def mode_accuracy(mode: str, **kwargs) -> dict:
    log_dict = {}
    if mode == "caption":
        output, full_labels = kwargs['output'], kwargs['full_labels']
        padding, topk = -100, (1, 5, 10)

        acc1, acc5, acc10 = accuracy(output[:, :-1, :], full_labels[:, 1:], padding=padding, topk=topk)
        log_dict["caption/acc1"] = acc1
        log_dict["caption/acc5"] = acc5
        log_dict["caption/acc10"] = acc10

    elif mode == "retrieval":
        logits_per_image = kwargs['logits_per_image']
        logits_per_text = kwargs['logits_per_text']

        caption_acc1, caption_acc5, caption_acc10 = contrastive_acc(logits_per_text, topk=(1, 5, 10))
        image_acc1, image_acc5, image_acc10 = contrastive_acc(logits_per_image, topk=(1, 5, 10))
        log_dict["retrieval/caption_acc1"] = caption_acc1
        log_dict["retrieval/caption_acc5"] = caption_acc5
        log_dict["retrieval/caption_acc10"] = caption_acc10
        log_dict["retrieval/image_acc1"] = image_acc1
        log_dict["retrieval/image_acc5"] = image_acc5
        log_dict["retrieval/image_acc10"] = image_acc10

    return log_dict




class ExpandChannels:
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if data.shape[0] != 1:
            raise ValueError(f"Expected input of shape [1, H, W], found {data.shape}")
        return torch.repeat_interleave(data, 3, dim=0)


def remap_to_uint8(array: np.ndarray, percentiles: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Remap values in input so the output range is :math:`[0, 255]`.

    Percentiles can be used to specify the range of values to remap.
    This is useful to discard outliers in the input data.

    :param array: Input array.
    :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
        Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
    :returns: Array with ``0`` and ``255`` as minimum and maximum values.
    """
    array = array.astype(float)
    if percentiles is not None:
        len_percentiles = len(percentiles)
        if len_percentiles != 2:
            message = 'The value for percentiles should be a sequence of length 2,' f' but has length {len_percentiles}'
            raise ValueError(message)
        a, b = percentiles
        if a >= b:
            raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
        if a < 0 or b > 100:
            raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
        cutoff: np.ndarray = np.percentile(array, percentiles)
        array = np.clip(array, *cutoff)
    array -= array.min()
    array /= array.max()
    array *= 255
    return array.astype(np.uint8)


def load_image(path: Path) -> Image.Image:
    if path.suffix in [".jpg", ".jpeg", ".png"]:
        image = io.imread(path)
    elif path.suffix == ".dcm":
        image = dicom.dcmread(path).pixel_array
    else:
        raise ValueError(f"Image type not supported, filename was: {path}")

    image = remap_to_uint8(image)
    return Image.fromarray(image).convert("L")


def preprocess_report(text: str) -> str:
    # Remove unnecessary and insensible parts
    text = re.sub(r"EXAMINATION:.*", "", text)  # Remove EXAMINATION line
    text = re.sub(r"WET READ:.*", "", text)  # Remove WET READ line
    text = re.sub(r"FINAL REPORT", "", text)  # Remove FINAL REPORT line
    text = re.sub(r"STUDY:.*", "", text)  # Remove STUDY line
    text = re.sub(r"COMPARISON:.*", "", text)  # Remove COMPARISON section
    text = re.sub(r"TECHNIQUE:.*", "", text)  # Remove TECHNIQUE section
    text = re.sub(r"_+", "_", text)  # Remove multiple underscores

    # Clean up excessive newlines and spaces
    text = re.sub(r"\s\s+", " ", text)
    text = re.sub("[^a-zA-Z0-9 :.,]", "", text)
    text = re.sub(r" +", " ", text)
    text = text.strip()
    return text