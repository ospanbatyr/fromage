import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Callable, Optional, Tuple, List, Dict
import numpy as np
import pydicom as dicom
from pathlib import Path
from skimage import io
import re
import string
import os
import os.path as osp
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True


class RETTokenCallback(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        ret_token_idx = pl_module.model.ret_token_idx

        for param in pl_module.model.model.input_embeddings.parameters():
            mask = torch.arange(param.grad.shape[0]) != ret_token_idx
            param.grad[mask,:] = 0.0


def create_callbacks(config, log_dir):
    checkpoints_path = osp.join(log_dir, 'checkpoints')
    config['checkpoint']['dirpath'] = checkpoints_path
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**config['checkpoint'])
    
    last_ckpt = osp.join(checkpoints_path, 'last.ckpt')
    last_ckpt = last_ckpt if osp.isfile(last_ckpt) else None
    ckpt_path = config['ckpt_path']['ckpt_path']

    if last_ckpt is not None and ckpt_path is not None:
        raise Exception('resume checkpoint passed (last.ckpt exists already)')

    ckpt_path = last_ckpt if ckpt_path is None else ckpt_path
    if ckpt_path is not None and not osp.isfile(ckpt_path):
        raise Exception('ckpt does not exist at {}'.format(ckpt_path))

    return [RETTokenCallback(), checkpoint_callback], ckpt_path


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


def contrastive_loss(logits):
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


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