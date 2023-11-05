import torch
import torch.nn as nn
from torch import optim
import pytorch_lightning as pl
from .model import Fromage
from .utils import mode_accuracy, retrieval_loss, get_logits


class Experiment(pl.LightningModule):
    def __init__(self, config=dict(), inference=False):
        super().__init__()
        self.automatic_optimization = False

        self.config = config
        self.inference = inference
        self.model = Fromage(self.device, inference, config)
        self.modes = ('caption', ) # , 'retrieval' disabled for debugging purposes
        self.save_hyperparameters(config)
    

    @property
    def model_config(self):
        return self.model.config


    def forward(self, cur_text, next_text, cur_pixels, next_pixels, mode):
        return self.model(cur_text, next_text, cur_pixels, next_pixels, mode=mode)


    def training_step(self, batch, batch_idx):
        cur_text, next_text, cur_pixels, next_pixels = batch
        opt = self.optimizers()
        opt.zero_grad()

        opt_config = self.config['optimizer']
        grad_acc_step = opt_config['grad_acc_step']

        losses = {f"{mode}_loss/train":0 for mode in self.modes}
        for mode in self.modes:
            opt.zero_grad()
            output, t2i_embs, i2t_embs, full_labels, last_logits = self.forward(cur_text, next_text, cur_pixels, next_pixels, mode=mode)
            loss = output.loss / grad_acc_step

            if mode == "retrieval":
                logits_per_image, logits_per_text = get_logits(t2i_embs, i2t_embs)
                loss += retrieval_loss(logits_per_image, logits_per_text) / grad_acc_step

            self.manual_backward(loss)

            if (batch_idx + 1) % grad_acc_step == 0:
                self.clip_gradients(opt, gradient_clip_val=opt_config['gradient_clip_val'], gradient_clip_algorithm="norm")
                opt.step()
                opt.zero_grad()

            if mode == "caption":
                log_acc_dict = mode_accuracy(mode=mode, output=output.logits, full_labels=full_labels)
            elif mode == "retrieval":
                log_acc_dict = mode_accuracy(mode=mode, logits_per_image=logits_per_image, logits_per_text=logits_per_text)

            losses[f"{mode}_loss/train"] = loss.item()

            log_acc_dict = {f"{k}/train": v for k, v in log_acc_dict.items()}
            self.log_dict(log_acc_dict, prog_bar=True)

        self.log_dict(losses, prog_bar=True)
        return losses


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pixels, text = batch

        losses = {f"{mode}_loss/val":0 for mode in self.modes}
        for mode in self.modes:
            output, t2i_embs, i2t_embs, full_labels, last_logits = self.forward(pixels, text, mode=mode)
            loss = output.loss

            if mode == "retrieval":
                logits_per_image, logits_per_text = get_logits(t2i_embs, i2t_embs)
                loss += retrieval_loss(t2i_embs, i2t_embs)

            if mode == "caption":
                log_acc_dict = mode_accuracy(mode=mode, output=output.logits, full_labels=full_labels)
            elif mode == "retrieval":
                log_acc_dict = mode_accuracy(mode=mode, logits_per_image=logits_per_image, logits_per_text=logits_per_text)

            losses[f"{mode}_loss/val"] = loss.item()

            log_acc_dict = {f"{k}/val": v for k, v in log_acc_dict.items()}
            self.log_dict(log_acc_dict, prog_bar=True)

        self.log_dict(losses, prog_bar=True)
        return losses


    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pixels, text = batch

        losses = {f"{mode}_loss/test":0 for mode in self.modes}
        for mode in self.modes:
            output, t2i_embs, i2t_embs, full_labels, last_logits = self.forward(pixels, text, mode=mode)
            loss = output.loss

            if mode == "retrieval":
                logits_per_image, logits_per_text = get_logits(t2i_embs, i2t_embs)
                loss += retrieval_loss(t2i_embs, i2t_embs)

            if mode == "caption":
                log_acc_dict = mode_accuracy(mode=mode, output=output.logits, full_labels=full_labels)
            elif mode == "retrieval":
                log_acc_dict = mode_accuracy(mode=mode, logits_per_image=logits_per_image, logits_per_text=logits_per_text)

            losses[f"{mode}_loss/test"] = loss.item()

            log_acc_dict = {f"{k}/test": v for k, v in log_acc_dict.items()}
            self.log_dict(log_acc_dict, prog_bar=True)

        self.log_dict(losses, prog_bar=True)
        return losses


    def configure_optimizers(self):
        opt_config = self.config['optimizer']
        opt_name = opt_config['algorithm']
        opt_params = opt_config['params']
        
        print("Total params:", sum(p.numel() for p in self.model.parameters()))

        # Model is loaded in fp16, all trainable params should be fp16
        if self.model.model.lm.lm_head:
            self.model.model.lm.lm_head.weight.data = self.model.model.lm.lm_head.weight.data.to(dtype=torch.float32)

        if self.model.model.lm.model.decoder.embed_tokens:
            self.model.model.lm.model.decoder.embed_tokens.weight.data = self.model.model.lm.model.decoder.embed_tokens.weight.data.to(dtype=torch.float32)
        elif self.model.model.lm.model.embed_tokens:
            self.model.model.lm.model.embed_tokens.weight.data = self.model.model.lm.model.embed_tokens.weight.data.to(dtype=torch.float32)

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())

        print("Trainable params:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        if opt_name == "AdamW":
            optimizer = optim.AdamW(parameters, **opt_params)
        elif opt_name == "Adam":
            optimizer = optim.Adam(parameters, **opt_params)
        else:
            raise NotImplementedError(f"Optimizer '{opt_name}' is not configured")

        return optimizer

