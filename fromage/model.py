import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, List, Dict
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from transformers import OPTForCausalLM, AutoTokenizer, AutoModelForCausalLM
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from .data import image_transform, RESIZE, CENTER_CROP_SIZE

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
        model_state_dict = torch.load("bin/biovil_backbone_2048.pt")
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
    def __init__(self, device, tokenizer, ret_token_idx, config):
        super().__init__()
        self.modes = ('caption', 'retrieval')
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        self.ret_token_idx = ret_token_idx
        self._init_language_model()
        self._init_image_encoder()
        self._init_mappers()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def _init_language_model(self) -> None:
        # create language model
        model_checkpoint = self.config['language_model']
        self.lm = AutoModelForCausalLM.from_pretrained(model_checkpoint)

        # freeze the language model
        for param in self.lm.parameters():
            param.requires_grad = False

        self.lm.eval()

        # resize token embeddings (as we added [RET] token) and 
        # get input embeddings as we will process information on embedding level, not index level 
        self.lm.resize_token_embeddings(len(self.tokenizer))
        self.input_embeddings = self.lm.get_input_embeddings()

        self.lm_embed_dim = self.input_embeddings.embedding_dim


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
        self.caption_mapping = nn.Linear(self.vm_embed_dim, self.lm_embed_dim)
        self.image_dropout = nn.Dropout(self.config['image_dropout'])

        if self.config['tie_mappers']:
            self.ret_i2t_mapping = self.caption_mapping
            self.ret_t2i_mapping = nn.Linear(self.lm_embed_dim, self.lm_embed_dim)
        else:
            self.shared_emb_dim = self.config['shared_emb_dim']
            self.ret_i2t_mapping = nn.Linear(self.vm_embed_dim, self.shared_emb_dim)
            self.ret_t2i_mapping = nn.Linear(self.lm_embed_dim, self.shared_emb_dim)

    
    def encode_images(self, pixel_values, mode):
        assert mode in self.modes, f'Mode must be in {str(self.modes)}, got {mode} instead'
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            img_embs = self.vm(pixel_values)

        if mode == "caption":
            img_embs = self.caption_mapping(img_embs)
            img_embs = self.image_dropout(img_embs)
        elif mode == "retrieval":
            img_embs = self.ret_i2t_mapping(img_embs)
            img_embs = self.image_dropout(img_embs)

        return img_embs

    
    def forward(self, pixel_values, text_inputs, mode):
        assert mode in self.modes, f'Mode must be in {str(self.modes)}, got {mode} instead'

        # if we are going to perform retrieval, we need to add
        # the [RET] token at the end of all text inputs
        if mode == "retrieval":
            new_text_inputs = []
            for i in range(len(text_inputs)):
                new_text_inputs.append(f'{text_inputs[i]}[RET]')
            text_inputs = tuple(new_text_inputs)

        max_length = self.config['max_length']
        text_inputs = self.tokenizer(
            text_inputs, return_tensors="pt", 
            padding="max_length", truncation=True, 
            max_length=max_length
        ).to(self.device)

        text_lens = text_inputs.attention_mask.sum(dim=1)

        # sometimes text length may be longer than max length, thus [RET] might not be present.
        # thus, we assert that last token id is [RET]
        if mode == "retrieval":
            for idx in range(len(text_inputs.input_ids)):
                if text_inputs.input_ids[idx][text_lens[idx]-1] != self.ret_token_idx:
                    text_inputs.input_ids[idx][text_lens[idx]-1] = self.ret_token_idx

        t2i_embs, i2t_embs = None, None

        if mode == "caption":
            img_embs = self.encode_images(pixel_values, mode=mode)
            img_embs = img_embs.unsqueeze(1)

            labels = text_inputs.input_ids
            text_embs = self.input_embeddings(labels)
            additional_mask = torch.ones(img_embs.shape[:2], dtype=torch.int64).to(self.device)
            attention_mask = torch.cat([additional_mask, text_inputs.attention_mask], dim=1)

            '''
            Q: Why do we use is `-100` for labels of image embeddings?**
            A: One way to handle this is to only train on the tag labels for the first subtoken of a split token. We can do this in 🤗 Transformers by setting the labels we wish to ignore to -100. 
               In the example above, if the label for @HuggingFace is 3 (indexing B-corporation), we would set the labels of `['@', 'hugging', '##face']` to `[3, -100, -100]`.
            Source: [https://huggingface.co/transformers/v4.4.2/custom_datasets.html](https://huggingface.co/transformers/v4.4.2/custom_datasets.html)
            '''

            full_labels = torch.full(img_embs.shape[:2], -100).to(self.device)
            full_labels = torch.cat([full_labels, labels], dim=1)

            input_embs = torch.cat([img_embs, text_embs], dim=1)
            output = self.lm(inputs_embeds=input_embs, attention_mask=attention_mask, labels=full_labels, output_hidden_states=True)
        
        elif mode == "retrieval":
            i2t_embs = self.encode_images(pixel_values, mode=mode)

            labels = text_inputs.input_ids
            text_embs = self.input_embeddings(labels)
            input_embs = text_embs

            output = self.lm(inputs_embeds=input_embs, attention_mask=text_inputs.attention_mask, labels=labels, output_hidden_states=True)

            last_hidden_state = output.hidden_states[-1]
            ret_embs = last_hidden_state[torch.arange(last_hidden_state.shape[0]), text_lens-1, :]
            t2i_embs = self.ret_t2i_mapping(ret_embs)

            i2t_embs = i2t_embs / i2t_embs.norm(dim=1, keepdim=True)
            t2i_embs = t2i_embs / t2i_embs.norm(dim=1, keepdim=True)

            i2t_embs = self.logit_scale.exp() * i2t_embs

        return output, t2i_embs, i2t_embs


        def generate(self, embeddings, max_len, temperature=0.0, top_p=1.0, filter_value=float("-inf")):
            bsz, seq_len, _ = embeddings.shape
            out = None
            output_embeddings = []
            output_logits = []

            with torch.no_grad():
                for i in range(max_len):
                    output = self.lm(inputs_embeds=embeddings, use_cache=False, output_hidden_states=True)
                    last_hidden_state = output.hidden_states[-1]
                    last_hidden_state = last_hidden_state[torch.arange(last_hidden_state.shape[0]), seq_len-1, :]
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

                    if top_p < 1.0:
                        assert top_p > 0, "0 < top_p <= 1 is not satisfied"
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        for j in range(sorted_indices.shape[0]):
                            indices_to_remove = sorted_indices[j, sorted_indices_to_remove[j, :]]
                            logits[j, indices_to_remove] = filter_value

                        token_weights = logits.exp()
                        next_token = torch.multinomial(token_weights, 1)

                    next_token = next_token.long().to(self.device)
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
    def __init__(self, device, config=dict()):
        super().__init__()
        self.device = device
        self.config = config
        self._init_tokenizer()
        self._init_inference_img_transform()
        self.model = FromageModel(device=self.device, tokenizer=self.tokenizer, ret_token_idx=self.ret_token_idx, config=self.model_config)
    

    @property
    def dataset_config(self):
        return self.config['dataset']

    
    @property
    def model_config(self):
        return self.config['model']

    
    def _init_inference_img_transform(self) -> None:
        resize = self.dataset_config['resize']
        center_crop_size = self.dataset_config['center_crop_size']
        self.img_transform = image_transform(resize=resize, center_crop_size=center_crop_size, train=False) 


    def _init_tokenizer(self) -> None:
        # create tokenizer
        model_checkpoint = self.model_config['language_model']
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, device=self.device)
        
        # add [RET] and <|image|> tokens and ensure that "[RET]" tokenization length is 1 
        tokenizer.add_special_tokens({"cls_token": "<|image|>"})
        tokenizer.add_tokens("[RET]")
        ret_id = tokenizer('[RET]', add_special_tokens=False).input_ids
        assert len(ret_id) == 1, "Failed to add [RET] token to tokenizer"

        ret_token_idx = ret_id[0]
        self.tokenizer = tokenizer
        self.ret_token_idx = ret_token_idx
        self.image_token = self.tokenizer.cls_token_id


    def __call__(self, images, tgt_tokens=None, generate=False, max_len=96, temperature=0.0, top_p=1.0, mode="caption", inference=False):
        if generate:
            return self.model.generate(embeddings=images, max_len=max_len, temperature=temperature, top_p=top_p)

        return self.model(pixel_values=images, text_inputs=tgt_tokens, mode=mode)


    def generate_for_images_and_texts(self, prompts: List, max_len=32, top_p=1.0, temperature=0.0):
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
            elif type(p) == str:
                tokens = self.model.tokenizer(p, add_special_tokens=True, return_tensors="pt")
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

        return_outputs = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]        
        return return_outputs