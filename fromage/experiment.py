import torch
import torch.nn as nn
from torch import optim
import pytorch_lightning as pl
from .model import Fromage
from .utils import contrastive_loss


class Experiment(pl.LightningModule):
    def __init__(self, config=dict()):
        super().__init__()
        self.automatic_optimization = False

        self.config = config
        self.model = Fromage(self.device, config)
        self.modes = ('caption', 'retrieval')
        self.save_hyperparameters(config)
    

    @property
    def model_config(self):
        return self.model.config


    def forward(self, pixels, text, mode):
        return self.model(pixels, text, mode=mode)


    def retrieval_loss(self, t2i_embs, i2t_embs):
        logits_per_image = i2t_embs @ t2i_embs.t()
        logits_per_text = logits_per_image.t()

        caption_loss = contrastive_loss(logits_per_text)
        image_loss = contrastive_loss(logits_per_image)
        
        return (caption_loss + image_loss) / 2.0


    def training_step(self, batch, batch_idx):
        pixels, text = batch
        opt = self.optimizers()
        opt_config = self.config['optimizer']

        losses = {f"{mode}_loss/train":0 for mode in self.modes}
        for mode in self.modes:
            output, i2t_embs, t2i_embs = self.forward(pixels, text, mode=mode)
            loss = output.loss

            if mode == "retrieval":
                loss += self.retrieval_loss(t2i_embs, i2t_embs)

            opt.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(opt, gradient_clip_val=opt_config['gradient_clip_val'], gradient_clip_algorithm="norm")

            opt.step()

            losses[f"{mode}_loss/train"] = loss.item()

        self.log_dict(losses, prog_bar=True)
        return losses


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pixels, text = batch

        losses = {f"{mode}_loss/val":0 for mode in self.modes}
        for mode in self.modes:
            output, i2t_embs, t2i_embs = self.forward(pixels, text, mode=mode)
            loss = output.loss

            if mode == "retrieval":
                loss += self.retrieval_loss(t2i_embs, i2t_embs)

            losses[f"{mode}_loss/val"] = loss.item()

        self.log_dict(losses, prog_bar=True)
        return losses

    def configure_optimizers(self):
        opt_config = self.config['optimizer']
        opt_name = opt_config['algorithm']
        opt_params = opt_config['params']
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())

        if opt_name == "AdamW":
            optimizer = optim.AdamW(parameters, **opt_params)
        elif opt_name == "Adam":
            optimizer = optim.Adam(parameters, **opt_params)
        else:
            raise NotImplementedError(f"Optimizer '{opt_name}' is not configured")

        return optimizer

