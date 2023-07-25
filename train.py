#!/usr/bin/env python

import os.path as osp

import pytorch_lightning as pl
import torch
import hydra
from omegaconf import OmegaConf

from fromage.data import MIMICDataModule as DataModule
from fromage.experiment import Experiment
from fromage.utils import create_callbacks, create_logger


CONFIG_DIR = osp.abspath(osp.join(__file__, "..", "config"))

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="train")
def main(config):
    config = OmegaConf.to_container(config)
    config = pl.utilities.parsing.AttributeDict(config)

    if "seed" in config:
        pl.seed_everything(config["seed"])
    print(config)

    dm = DataModule(config)
    logger = create_logger(config)
    callbacks = None

    logger_conf = config["logger"]
    if logger is not None and logger_conf["version"] != "debug":
        callbacks, ckpt_path = create_callbacks(config, logger_conf["save_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment = Experiment(device, config)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **config["trainer"])
    
    trainer.fit(experiment, datamodule=dm)


if __name__ == "__main__":
    main()