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
import argparse

# Parse the max generation lengths
parser = argparse.ArgumentParser(description="Set max lengths for VQA closed and open questions.")

parser.add_argument("--max-len-vqa-closed", type=int, default=1, help="Maximum length for VQA closed questions")
parser.add_argument("--max-len-vqa-open", type=int, default=16, help="Maximum length for VQA open questions")

args = parser.parse_args()

max_len_vqa_closed = args.max_len_vqa_closed
max_len_vqa_open = args.max_len_vqa_open


# VARIABLES
ckpt_path = "../logs/checkpoints/lm_gen_vis_med_mistral_rerun2/last.ckpt"
config_path = "../config/train-untied_lm_gen_vis_med.yaml"
dataset_path = "../data/datasets/VQA_RAD"
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

transform = cxr_image_transform(resize=512, center_crop_size=480, train=False) 
dataset_closed = VQA_RADDataset(dataset_path, transform, 'closed')
dataset_open = VQA_RADDataset(dataset_path, transform, 'open')

print("VQA-RAD Closed Length: ", dataset_closed.get_len())
print("VQA-RAD Open Length: ", dataset_open.get_len())

right_answers = 0
total_answers = 0

vqa_rad_closed_cls = ["yes", "no"]
for i, idx in enumerate(tqdm(dataset_closed)):
    img, q, ans = idx 
    with torch.inference_mode() as inf_mode, torch.autocast(device_type="cuda") as cast:
        model.eval()
        prompts = [idx[0], str("Question: " + idx[1] + " Answer the question with only yes or no: ")]
        model_ans = vqa_rad_closed_cls[model.classification_for_eval(prompts, vqa_rad_closed_cls)] # top_p=0.9, temperature=0.5
        model_ans = model_ans.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
        
        print(model_ans, ans)
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
        prompts = [idx[0], str("Question: " + idx[1] + " Answer the question using a single word or phrase: ")] 
        max_bleu_score = 0
        for _ in range(5): # try 5 times, get the best score of those 5 times
            try:
                model_ans = model.generate_for_images_and_texts(prompts, max_len=max_len_vqa_open, top_p=0.9, temperature=0.5)    
                bleu_score = bleu_metric.compute(predictions=[model_ans], references=[ans]).get('bleu')
                max_bleu_score = max(max_bleu_score, bleu_score)
            except:
                pass

    total += 1

print(f"Open Question BLEU Score : {total_bleu_score} / {total} = {(total_bleu_score / total) * 100}")
