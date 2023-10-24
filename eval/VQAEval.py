#!/usr/bin/env python
# coding: utf-8

# # Tutorial & Evaluation

# In[1]:


import sys
import torch
import yaml
from tqdm import tqdm

from fromage.vqa_dataset import VQA_RADDataset
from fromage.model import Fromage, FromageModel
from fromage.experiment import Experiment
from fromage.data import MIMICDataset, cxr_image_transform
from fromage.utils import preprocess_report
from evaluate import load # if throws error, please run the following command "pip instal evaluate"


# In[2]:


## config
ckpt_path = "../logs/checkpoints/lm_gen_vis_med_mistral_rerun2/last.ckpt"
config_path = "../config/train-untied_lm_gen_vis_med.yaml"
dataset_path = "../data/datasets/VQA_RAD"
MAX_LEN = int(sys.argv[1])


# In[3]:


transform = cxr_image_transform(resize=512, center_crop_size=480, train=False) 
dataset = VQA_RADDataset(dataset_path, transform)


# In[4]:


dataset[0] # returns image, question, answer


# In[ ]:


with open(config_path) as file:
    config = yaml.safe_load(file)

device = "cuda"

model = Experiment(config)
model = model.load_from_checkpoint(ckpt_path)
model = model.model.to(device)
model.device = device

# In[ ]:




img, question, answer = dataset[0]
prompt = str("Question: " + question + " Answer: ")
print("Prompt: ", prompt)

model.eval()

print("Total params:", sum(p.numel() for p in model.parameters()))

# for name, p in model.named_parameters():
#     print(name, p.requires_grad, p.data.dtype)

parameters = filter(lambda p: p.requires_grad, model.parameters())

print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

with torch.inference_mode() as inf_mode, torch.autocast(device_type="cuda") as cast:
    prompts = [img, prompt] 
    print("Model Answer: ", model.generate_for_images_and_texts(prompts, max_len=MAX_LEN)) # top_p=0.9, temperature=0.5
    
print("Correct Answer: ", answer)


# # Evaluation

# In[ ]:


transform = cxr_image_transform(resize=512, center_crop_size=480, train=False) 
dataset_closed = VQA_RADDataset(dataset_path, transform, 'closed')
dataset_open = VQA_RADDataset(dataset_path, transform, 'open')

print(dataset_closed.get_len())
print(dataset_open.get_len())


# ## Closed dataset: accuracy

# In[ ]:


import string 

right_answers = 0
total_answers = 0

def get_model_response(prompts, max_len):
    model_ans_full = model.generate_for_images_and_texts(prompts, max_len=max_len) # top_p=0.9, temperature=0.5
    model_ans = model_ans_full.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    return model_ans

for i, idx in enumerate(tqdm(dataset_closed)):
    img, q, ans = idx 
    with torch.inference_mode() as inf_mode, torch.autocast(device_type="cuda") as cast:
        model.eval()
        prompts = [idx[0], str("Question: " + idx[1] + "Answer (Yes or No): ")] 
        model_ans = get_model_response(prompts, max_len=MAX_LEN)
        if i % 50 == 1:
            print(model_ans.lower(), ans.lower())
        if model_ans.lower() == ans.lower():
            right_answers += 1
        total_answers += 1        

print(right_answers, '/', total_answers)
print((right_answers/total_answers)*100, '% correct')

# ## Open dataset: Bleu score

# In[ ]:


exact_match_metric = load("bleu")


# In[ ]:


total_bleu_score = 0
total = 0

for idx in tqdm(dataset_open):
    img, q, ans = idx 
    with torch.inference_mode() as inf_mode, torch.autocast(device_type="cuda") as cast:
        model.eval()
        prompts = [idx[0], str("Question: " + idx[1] + " Answer: ")] 
        model_ans_full = model.generate_for_images_and_texts(prompts, max_len=MAX_LEN) # top_p=0.9, temperature=0.5    
        current_bleu_scores = []
        for _ in range(4): # try 5 times, get the best score of those 5 times
            try:
                bleu_score = exact_match_metric.compute(predictions=[model_ans_full], references=[ans]).get('bleu')
                current_bleu_scores.append(bleu_score)
            except:
                pass
        if len(current_bleu_scores) > 1:
            total_bleu_score += max(current_bleu_scores) # you can also take the average
        total += 1
        
print(total_bleu_score, '/', total)
print("bleu score: ", total_bleu_score/total)

