import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from transformers import OPTForCausalLM, AutoTokenizer, AutoModelForCausalLM
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from .data import cxr_image_transform, coco_image_transform
from .utils import load_image, save_model_path

BIN_DIR = osp.abspath(osp.join(__file__, "..", "..", "bin"))


VM_EMBED_DIMS = {
    'biovil': 2048,
    'resnet-50-imagenet': 2048
}


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = self._get_feature_extractor()
    
    def _get_feature_extractor(self):
        self._return_nodes = {'avgpool': 'avgpool'}
        return create_feature_extractor(self.model, return_nodes=self._return_nodes)

    def forward(self, x):
        features = self.feature_extractor(x)["avgpool"]
        features = features.squeeze()
        return features    


class BioViL(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50()
        self._initialize_resnet()
        self.feature_extractor = self._get_feature_extractor()
        
    def _initialize_resnet(self):
        model_state_dict = torch.load(osp.join(BIN_DIR, "biovil_backbone_2048.pt"))
        self.model.load_state_dict(model_state_dict)
    
    def _get_feature_extractor(self):
        self._return_nodes = {'avgpool': 'avgpool'}
        return create_feature_extractor(self.model, return_nodes=self._return_nodes)

    def forward(self, x):
        features = self.feature_extractor(x)["avgpool"]
        features = features.squeeze()
        return features


def get_vision_model(vm_name):
    if vm_name == "biovil":
        return BioViL()
    elif vm_name == "resnet-50-imagenet":
        return ResNet50()
    
    raise NotImplementedError(f"{vm_name} is not configured")


def get_vm_embed_dim(vm_name):
    return VM_EMBED_DIMS[vm_name]


class FromageModel(nn.Module):
    def __init__(self, tokenizer, ret_token_idx, inference, config, general_config):
        super().__init__()
        self.modes = ('caption', 'retrieval')
        self.config = config
        self.general_config = general_config
        self.inference = inference
        self.tokenizer = tokenizer
        self.ret_token_idx = ret_token_idx
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self._init_language_model()
        self._init_image_encoder()
        self._init_mappers()
        self._load_model()


    def _init_language_model(self) -> None:
        # create language model
        lm_config = self.config['language_model']

        if self.inference:
            self.lm = AutoModelForCausalLM.from_pretrained(lm_config['name'], **lm_config["inference"])
        else:
            self.lm = AutoModelForCausalLM.from_pretrained(lm_config['name'], torch_dtype=torch.float16)

        # freeze the language model
        for name, param in self.lm.named_parameters():
            param.requires_grad = False
            
        # resize token embeddings (as we added [RET] token) and 
        # get input embeddings as we will process information on embedding level, not index level 
        self.lm.resize_token_embeddings(len(self.tokenizer))
        self.input_embeddings = self.lm.get_input_embeddings()

        self.lm_embed_dim = self.input_embeddings.embedding_dim

        self.lm.eval()


    def _init_image_encoder(self) -> None:
        # create image encoder
        vm_name = self.config['vision_model']
        
        self.vm = get_vision_model(vm_name)
        self.vm_embed_dim = get_vm_embed_dim(vm_name)

        # freeze image encoder
        for param in self.vm.parameters():
            param.requires_grad = False

        self.vm.eval()


    def _init_mappers(self) -> None:
        self.num_img_tokens = self.config['num_img_tokens']

        self.caption_mapping = nn.Linear(self.vm_embed_dim, self.lm_embed_dim * self.num_img_tokens) # VM_EMBED_DIM x LM_EMBED_DIM
        self.image_dropout = nn.Dropout(self.config['image_dropout'])

        if self.config['tie_mappers']:
            self.ret_i2t_mapping = self.caption_mapping
            self.ret_t2i_mapping = nn.Linear(self.lm_embed_dim, self.lm_embed_dim * self.num_img_tokens)
        else:
            self.shared_emb_dim = self.config['shared_emb_dim']
            self.ret_i2t_mapping = nn.Linear(self.vm_embed_dim, self.shared_emb_dim)
            self.ret_t2i_mapping = nn.Linear(self.lm_embed_dim, self.shared_emb_dim)

    
    def _load_model(self):
        if not self.inference:
            return

        weight_path = save_model_path(self.general_config)
        print(str(self.load_state_dict(torch.load(weight_path, map_location=self.logit_scale.device), strict=False)))


    def encode_images(self, pixel_values, mode):
        if len(pixel_values.shape) == 5: # means we have multiple images
            bsz, img_cnt, ch, h, w = pixel_values.shape
            pixel_values = pixel_values.view(bsz * img_cnt, ch, h, w)
        else:
            bsz, ch, h, w = pixel_values.shape
            img_cnt = 1

        assert mode in self.modes, f'Mode must be in {str(self.modes)}, got {mode} instead'
        pixel_values = pixel_values.to(self.logit_scale.device)

        with torch.no_grad():
            img_embs = self.vm(pixel_values)

        if mode == "caption":
            img_embs = self.caption_mapping(img_embs)
            img_embs = self.image_dropout(img_embs)
        elif mode == "retrieval":
            img_embs = self.ret_i2t_mapping(img_embs)
            img_embs = self.image_dropout(img_embs)

        bsz_n_img_cnt, emb_dim = img_embs.shape
        img_embs = img_embs.view(bsz, img_cnt, emb_dim)

        return img_embs

    
    def forward(self, cur_pixel_values, next_pixel_values, cur_text_inputs, next_text_inputs, mode):
        assert mode in self.modes, f'Mode must be in {str(self.modes)}, got {mode} instead'

        # if we are going to perform retrieval, we need to add
        # the [RET] token at the end of all text inputs
        if mode == "retrieval":
            def add_ret_token(text_inputs):
                new_text_inputs = []
                for i in range(len(text_inputs)):
                    new_text_inputs.append(f'{text_inputs[i]}[RET]')
                return tuple(new_text_inputs)

            cur_text_inputs = add_ret_token(cur_text_inputs)
            next_text_inputs = add_ret_token(next_text_inputs) if next_text_inputs else None


        max_length = self.config['max_length']
        def tokenize_text(text_inputs):
            return self.tokenizer(
                text_inputs, return_tensors="pt", 
                padding="max_length", truncation=True, 
                max_length=max_length
            ).to(self.logit_scale.device)

        cur_text_inputs = tokenize_text(cur_text_inputs)
        next_text_inputs = tokenize_text(next_text_inputs) if next_text_inputs is not None else None

        cur_text_lens = cur_text_inputs.attention_mask.sum(dim=1)
        next_text_lens = next_text_inputs.attention_mask.sum(dim=1) if next_text_inputs is not None else None
        # sometimes text length may be longer than max length, thus [RET] might not be present.
        # thus, we assert that last token id is [RET]
        if mode == "retrieval":
            def ensure_ret_token(text_inputs):
                for idx in range(len(text_inputs.input_ids)):
                    if text_inputs.input_ids[idx][text_lens[idx]-1] != self.ret_token_idx:
                        text_inputs.input_ids[idx][text_lens[idx]-1] = self.ret_token_idx
                return text_inputs

            cur_text_inputs = ensure_ret_token(cur_text_inputs)
            next_text_inputs = ensure_ret_token(next_text_inputs) if next_text_inputs is not None else None


        t2i_embs, i2t_embs, last_logits = None, None, None

        if mode == "caption":
            cur_img_embs = self.encode_images(cur_pixel_values, mode=mode)
            next_img_embs = self.encode_images(next_pixel_values, mode=mode) if next_pixel_values.nelement() != 0 else None

            def prepare_text_args(text_inputs, img_embs):
                labels = text_inputs.input_ids
                text_embs = self.input_embeddings(labels)
                additional_mask = torch.ones(img_embs.shape[:2], dtype=torch.int64).to(self.logit_scale.device)
                attention_mask = torch.cat([additional_mask, text_inputs.attention_mask], dim=1)
                full_labels = torch.full(img_embs.shape[:2], -100).to(self.logit_scale.device)
                full_labels = torch.cat([full_labels, labels], dim=1)
                return text_embs, attention_mask, full_labels

            cur_text_embs, cur_attention_mask, cur_full_labels = prepare_text_args(cur_text_inputs, cur_img_embs)   

            '''
            Q: Why do we use is `-100` for labels of image embeddings?**
            A: One way to handle this is to only train on the tag labels for the first subtoken of a split token. We can do this in 🤗 Transformers by setting the labels we wish to ignore to -100. 
               In the example above, if the label for @HuggingFace is 3 (indexing B-corporation), we would set the labels of `['@', 'hugging', '##face']` to `[3, -100, -100]`.
            Source: [https://huggingface.co/transformers/v4.4.2/custom_datasets.html](https://huggingface.co/transformers/v4.4.2/custom_datasets.html)
            '''

            if next_text_inputs is not None and next_img_embs is not None:
                next_text_embs, next_attention_mask, next_full_labels = prepare_text_args(next_text_inputs, next_img_embs)
                input_embs = [cur_img_embs, cur_text_embs, next_img_embs, next_text_embs]
                attention_mask = torch.cat([cur_attention_mask, next_attention_mask], dim=1)
                full_labels = torch.cat([cur_full_labels, next_full_labels], dim=1)
            else:
                input_embs = [cur_img_embs, cur_text_embs]
                attention_mask = cur_attention_mask
                full_labels = cur_full_labels

            input_embs = torch.cat(input_embs, dim=1)
            output = self.lm(inputs_embeds=input_embs, attention_mask=attention_mask, labels=full_labels, output_hidden_states=True)
        
        elif mode == "retrieval":
            i2t_embs = self.encode_images(pixel_values, mode=mode)

            full_labels = text_inputs.input_ids
            text_embs = self.input_embeddings(full_labels)
            input_embs = text_embs

            output = self.lm(inputs_embeds=input_embs, attention_mask=text_inputs.attention_mask, labels=full_labels, output_hidden_states=True)

            last_hidden_state = output.hidden_states[-1] # decoder-only, 12 decoder layer, take last layer's hidden state
            ret_embs = last_hidden_state[torch.arange(last_hidden_state.shape[0]), text_lens-1, :]
            t2i_embs = self.ret_t2i_mapping(ret_embs)

            last_logits = output.logits[torch.arange(last_hidden_state.shape[0]), text_lens-2, :]

            i2t_embs = i2t_embs / i2t_embs.norm(dim=1, keepdim=True)
            t2i_embs = t2i_embs / t2i_embs.norm(dim=1, keepdim=True)

            i2t_embs = self.logit_scale.exp() * i2t_embs

        return output, t2i_embs, i2t_embs, full_labels, last_logits


    def perplexity(self, prompt_embeddings, expected_tok_ids):
        bsz, seq_len, dim = prompt_embeddings.shape
        ppl = 0
        embeddings = prompt_embeddings

        with torch.no_grad():
            for tok_id in expected_tok_ids:
                output = self.lm(inputs_embeds=embeddings, use_cache=False, output_hidden_states=True)
                logits = output.logits[:,-1,:]
                probs = torch.softmax(logits, dim=-1)

                tok_id = tok_id.unsqueeze(0)
                cur_tok_prob = probs[:, tok_id]

                ppl += torch.log(cur_tok_prob)
                next_embedding = self.input_embeddings(tok_id).unsqueeze(0)
                embeddings = torch.cat([embeddings, next_embedding], dim=1)

        ppl = torch.exp(-ppl)
        return ppl.item()


    def generate(self, embeddings, max_len, temperature=0.0, top_p=1.0, filter_value=float("-inf")):
        bsz, seq_len, _ = embeddings.shape
        out = None
        output_embeddings = []
        output_logits = []

        with torch.no_grad():
            for i in range(max_len):
                output = self.lm(inputs_embeds=embeddings, use_cache=False, output_hidden_states=True)
                last_hidden_state = output.hidden_states[-1]
                last_hidden_state = last_hidden_state[torch.arange(last_hidden_state.shape[0]), seq_len-1, :] # 
                last_hidden_state = self.ret_t2i_mapping(last_hidden_state)
                last_embedding = last_hidden_state / last_hidden_state.norm(dim=-1, keepdim=True)

                output_embeddings.append(last_embedding)
                logits = output.logits[:,-1,:]
                output_logits.append(logits)

                if temperature == 0.0:
                    if top_p != 1.0:
                        assert False, "top_p cannot be set in greedy decoding"
                    next_token = torch.argmax(logits, keepdim=True, dim=-1)
                else:
                    logits = logits / temperature

                if top_p < 1.0: # top p sampling. 
                    assert top_p > 0, "0 < top_p <= 1 is not satisfied"
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True) # 
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    for j in range(sorted_indices.shape[0]):
                        indices_to_remove = sorted_indices[j, sorted_indices_to_remove[j, :]]
                        logits[j, indices_to_remove] = filter_value

                    token_weights = logits.exp()
                    next_token = torch.multinomial(token_weights, 1)

                next_token = next_token.long().to(self.logit_scale.device)

                if out is not None:
                    out = torch.cat([out, next_token], dim=1)
                else:
                    out = next_token

                next_embedding = self.input_embeddings(next_token)
                embeddings = torch.cat([embeddings, next_embedding], dim=1)

                if (self.tokenizer.eos_token_id and (next_token == self.tokenizer.eos_token_id).all()):
                    break

        return out, output_embeddings, output_logits

        
        def train(self, mode=True):
            super(FromageModel, self).train(mode=mode)
            self.lm.eval()
            self.vm.eval()


class Fromage(nn.Module):
    def __init__(self, device, inference, config=dict()):
        super().__init__()
        self.config = config
        self.device = device
        self.inference = inference
        self._init_tokenizer()
        self._init_inference_img_transform()
        self.model = FromageModel(tokenizer=self.tokenizer, ret_token_idx=self.ret_token_idx, inference=inference, config=self.model_config, general_config=self.config)


    @property
    def dataset_config(self):
        return self.config['dataset']

    
    @property
    def model_config(self):
        return self.config['model']

    
    def _init_inference_img_transform(self) -> None:
        if self.dataset_config["name"] == "COCO":
            self.img_transform = coco_image_transform(train=False)
        elif self.dataset_config["name"] == "MIMIC-CXR-JPG":
            resize = self.dataset_config['resize']
            center_crop_size = self.dataset_config['center_crop_size']
            self.img_transform = cxr_image_transform(resize=resize, center_crop_size=center_crop_size, train=False) 


    def _init_tokenizer(self) -> None:
        # create tokenizer
        model_checkpoint = self.model_config['language_model']['name']

        if "opt-125m" in model_checkpoint:
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=False, device=self.device)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False, device=self.device)
        
        # add [RET] and <|image|> tokens and ensure that "[RET]" tokenization length is 1 
        tokenizer.add_special_tokens({"cls_token": "<|image|>"})
        tokenizer.add_tokens("[RET]")
        ret_id = tokenizer('[RET]', add_special_tokens=False).input_ids
        assert len(ret_id) == 1, "Failed to add [RET] token to tokenizer"

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ret_token_idx = ret_id[0]
        self.tokenizer = tokenizer
        self.ret_token_idx = ret_token_idx
        self.image_token = self.tokenizer.cls_token_id


    def __call__(self, cur_images, next_images=None, cur_tgt_tokens=None, next_tgt_tokens=None, generate=False, max_len=96, temperature=0.0, top_p=1.0, mode="caption", inference=False):
        if generate:
            return self.model.generate(embeddings=cur_images, max_len=max_len, temperature=temperature, top_p=top_p)

        return self.model(cur_images, next_images, cur_tgt_tokens, next_tgt_tokens, mode=mode)


    def classification_for_eval(self, prompts: List, classes: List, add_special_tokens=True):
        input_embs = []
        input_ids = []

        add_bos = True
        for i, p in enumerate(prompts):
            if isinstance(p, Path):
                img = load_image(p)
                pixel_values = self.img_transform(img)
                pixel_values = pixel_values[None, ...]
                vis_emb = self.model.encode_images(pixel_values, mode="caption")
                vis_emb = vis_emb.unsqueeze(0).unsqueeze(0)
                input_embs.append(vis_emb)
            elif isinstance(p, torch.Tensor):
                pixel_values = p[None, ...]
                vis_emb = self.model.encode_images(pixel_values, mode="caption")
                vis_emb = vis_emb.unsqueeze(0).unsqueeze(0)
                input_embs.append(vis_emb)
            elif type(p) == str:
                tokens = self.model.tokenizer(p, add_special_tokens=add_special_tokens, return_tensors="pt")
                text_ids = tokens.input_ids.to(self.device)
                if not add_bos:
                    text_ids = text_ids[:, 1:]
                else:
                    add_bos = False

                text_embs = self.model.input_embeddings(text_ids)
                input_embs.append(text_embs)
                input_ids.append(text_ids)

        input_embs = torch.cat(input_embs, dim=1)
        input_ids = torch.cat(input_ids, dim=1)

        min_ppl, min_ppl_idx = float("inf"), -1
        for cls_idx, cls_name in enumerate(classes):
            expected_tokens = self.model.tokenizer(cls_name, add_special_tokens=False, return_tensors="pt")
            expected_tok_ids = expected_tokens.input_ids[0].to(self.device)

            curr_ppl = self.model.perplexity(input_embs, expected_tok_ids)

            if curr_ppl < min_ppl:
                min_ppl = curr_ppl
                min_ppl_idx = cls_idx

        return classes[min_ppl_idx]


    def generate_for_images_and_texts(self, prompts: List, max_len=32, top_p=1.0, temperature=0.0, add_special_tokens=True):
        input_embs = []
        input_ids = []

        add_bos = True

        for i, p in enumerate(prompts):
            if isinstance(p, Path):
                img = load_image(p)
                pixel_values = self.img_transform(img)
                pixel_values = pixel_values[None, ...]
                vis_emb = self.model.encode_images(pixel_values, mode="caption")
                vis_emb = vis_emb.unsqueeze(0).unsqueeze(0)
                input_embs.append(vis_emb)
            elif isinstance(p, torch.Tensor):
                pixel_values = p[None, ...]
                vis_emb = self.model.encode_images(pixel_values, mode="caption")
                vis_emb = vis_emb.unsqueeze(0).unsqueeze(0)
                input_embs.append(vis_emb)
            elif type(p) == str:
                tokens = self.model.tokenizer(p, add_special_tokens=add_special_tokens, return_tensors="pt")
                text_ids = tokens.input_ids.to(self.device)
                if not add_bos:
                    text_ids = text_ids[:, 1:]
                else:
                    add_bos = False

                text_embs = self.model.input_embeddings(text_ids)
                input_embs.append(text_embs)
                input_ids.append(text_ids)
            else:
                assert False, "Prompt type can only be Path for images or string for text"

        input_embs = torch.cat(input_embs, dim=1)
        input_ids = torch.cat(input_ids, dim=1)

        generated_ids, generated_embeddings, _ = self.model.generate(input_embs, max_len, temperature=temperature, top_p=top_p)

        return_outputs = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]        
        return return_outputs