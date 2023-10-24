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
import string 


# VARIABLES
ckpt_path = "../logs/checkpoints/lm_gen_vis_med_mistral_rerun2/last.ckpt"
config_path = "../config/train-untied_lm_gen_vis_med.yaml"
dataset_path = "../data/datasets/VQA_RAD"
MAX_LEN_VQA_CLOSED = int(sys.argv[1])
MAX_LEN_VQA_OPEN = int(sys.argv[2])
bleu_metric = load("bleu")

# LOAD MODEL
with open(config_path) as file:
    config = yaml.safe_load(file)

device = "cuda"
model = Experiment(config)
model = model.load_from_checkpoint(ckpt_path)
model = model.model.to(device)
model.device = device

model.eval()

dataset_closed = VQA_RADDataset(dataset_path, transform, 'closed')
dataset_open = VQA_RADDataset(dataset_path, transform, 'open')

print("VQA-RAD Closed Length: ", dataset_closed.get_len())
print("VQA-RAD Open Length: ", dataset_open.get_len())

right_answers = 0
total_answers = 0

for i, idx in enumerate(tqdm(dataset_closed)):
    img, q, ans = idx 
    with torch.inference_mode() as inf_mode, torch.autocast(device_type="cuda") as cast:
        model.eval()
        prompts = [idx[0], str("Question: " + idx[1] + "Answer (Yes or No): ")] 
        model_ans = model.generate_for_images_and_texts(prompts, max_len=MAX_LEN_VQA_CLOSED) # top_p=0.9, temperature=0.5
        model_ans = model_ans.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
        
        if i % 50 == 1:
            print(model_ans.lower(), ans.lower())
        if model_ans.lower() == ans.lower():
            right_answers += 1
        total_answers += 1        

print("VQA-RAD Evaluation")
print("="*30)
print(f"Closed (Yes/No) Question Accuracy: {right_answers} / {total_answers} = {(right_answers / total_answers)*100}")

total_bleu_score = 0
total = 0

for idx in tqdm(dataset_open):
    img, q, ans = idx 
    with torch.inference_mode() as inf_mode, torch.autocast(device_type="cuda") as cast:
        model.eval()
        prompts = [idx[0], str("Question: " + idx[1] + " Answer: ")] 
        model_ans = model.generate_for_images_and_texts(prompts, max_len=MAX_LEN_VQA_OPEN) # top_p=0.9, temperature=0.5    
        max_bleu_score = 0
        for _ in range(5): # try 5 times, get the best score of those 5 times
            try:
                bleu_score = bleu_metric.compute(predictions=[model_ans], references=[ans]).get('bleu')
                max_bleu_score = max(max_bleu_score, bleu_score)
            except:
                pass

    total += 1

print(f"Open Question BLEU Score : {total_bleu_score} / {total} = {(total_bleu_score / total) * 100}")
