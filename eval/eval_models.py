import sys
import torch
import yaml
import string 
import argparse
from tqdm import tqdm
import os.path as osp

from fromage.vqa_dataset import VQA_RADDataset
from fromage.imgcls_dataset import RSNAPneumoniaDataset, COVIDDataset
from fromage.model import Fromage, FromageModel
from fromage.experiment import Experiment
from fromage.data import MIMICDataset, cxr_image_transform
from fromage.utils import preprocess_report
from evaluate import load # if throws error, please run the following command "pip instal evaluate"


# Parse the max generation lengths
parser = argparse.ArgumentParser(description="Set max lengths for VQA closed and open questions.")

parser.add_argument("--max-len-vqa-open", type=int, default=10, help="Maximum length for VQA open questions")
parser.add_argument("--ckpt-dir", type=str, default="../logs/checkpoints", help="Default folder for checkpoints")
parser.add_argument("--dataset-dir", type=str, default="../data/datasets", help="Default folder for datasets")
parser.add_argument("--ckpt-name", type=str, required=True, help="Checkpoint name")
parser.add_argument("--eval-datasets", type=lambda s: s.split(","), default="covid,vqarad,rsna", help="Datasets to be used for evaluation")

args = parser.parse_args()
max_len_vqa_open = args.max_len_vqa_open
ckpt_dir = args.ckpt_dir
dataset_dir = args.dataset_dir
ckpt_name = args.ckpt_name
eval_datasets = args.eval_datasets

# VARIABLES
ckpt_path = osp.join(ckpt_dir, ckpt_name, "last.ckpt")
config_path = osp.join(ckpt_dir, ckpt_name, "config.yaml")

vqa_dataset_path = osp.join(dataset_dir, "VQA_RAD")
rsna_dataset_path = osp.join(dataset_dir, "RSNA")
covid_dataset_path = osp.join(dataset_dir, "COVID_QU_EX")
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

# DATASETS
transform = cxr_image_transform(resize=512, center_crop_size=480, train=False) 

# VQA-RAD EVALUATION
def VQARAD_evaluation():
    closed_VQARAD_dataset = VQA_RADDataset(vqa_dataset_path, transform, 'closed')
    open_VQARAD_dataset = VQA_RADDataset(vqa_dataset_path, transform, 'open')
    right_answers = 0
    total_answers = 0

    vqa_rad_closed_cls = ["yes", "no"]
    for i, idx in enumerate(tqdm(closed_VQARAD_dataset)):
        img, q, ans = idx 
        with torch.inference_mode() as inf_mode, torch.autocast(device_type="cuda") as cast:
            model.eval()
            prompts = [idx[0], str("Question: " + idx[1] + " Answer the question with only yes or no: ")]
            model_ans = model.classification_for_eval(prompts, vqa_rad_closed_cls) # top_p=0.9, temperature=0.5

            if model_ans.lower() == ans.lower():
                right_answers += 1
            total_answers += 1        

    print(f"VQA-RAD Closed Question Accuracy: {right_answers} / {total_answers} = {(right_answers / total_answers)*100}")

    total_bleu_score = 0
    total = 0

    for idx in tqdm(open_VQARAD_dataset):
        img, q, ans = idx 
        with torch.inference_mode() as inf_mode, torch.autocast(device_type="cuda") as cast:
            model.eval()
            """<s>[INST] You are an AI assistant specialized in chest X-ray radiology question answering using a single word or a few words. Use only a single word or a few words to answer the questions. [/INST] Understood.</s>[INST] """

            prompts = [
                idx[0], "Question: " + idx[1] + " Answer using a single word or a few words: "
            ]

            max_bleu_score = 0
            for _ in range(5): # try 5 times, get the best score of those 5 times
                try:
                    model_ans = model.generate_for_images_and_texts(prompts, max_len=max_len_vqa_open, top_p=0.9, temperature=0.5, add_special_tokens=False)
                    model_ans = model_ans.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
                    bleu_score = bleu_metric.compute(predictions=[model_ans.lower()], references=[[ans.lower()]]).get('bleu')
                    max_bleu_score = max(max_bleu_score, bleu_score)
                except:
                    pass

            total_bleu_score += max_bleu_score

        total += 1

    print(f"VQA-RAD Open Question BLEU: {total_bleu_score} / {total} = {(total_bleu_score / total) * 100}")

# CLASSIFICATION

def RSNA_evaluation():
    # RSNA EVALUATION
    RSNAdataset = RSNAPneumoniaDataset(rsna_dataset_path, transform)
    right_answers = 0
    total_answers = 0

    rsna_classes = ["yes", "no"]

    for idx in tqdm(RSNAdataset):
        img, ans = idx 
        with torch.inference_mode() as inf_mode, torch.autocast(device_type="cuda") as cast:
            model.eval()
            prompts = [idx[0], "Question: Is pneumonia present in this chest x-ray image? Answer (yes or no): "] 
            model_ans = model.classification_for_eval(prompts, rsna_classes) # top_p=0.9, temperature=0.5
            
            if model_ans.lower() == ans.lower():
                right_answers += 1

            total_answers += 1  

    print(f"RSNA Classification Accuracy: {right_answers} / {total_answers} = {(right_answers / total_answers)*100}")

def COVID_evaluation():
    # COVID EVALUATION
    COVIDdataset = COVIDDataset(covid_dataset_path, transform)
    right_answers = 0
    total_answers = 0

    covid_classes = ["A", "B", "C"]

    for idx in tqdm(COVIDdataset):
        img, ans = idx 
        with torch.inference_mode() as inf_mode, torch.autocast(device_type="cuda") as cast:
            model.eval()
            """<s>[INST] You are an AI assistant specialized in chest X-ray radiology image classification. Here are the image classes and teir definitions:
            normal: The absence of diseases and infirmity, indicating the structure is normal.
            pneumonia: An inflammatory condition of the lung primarily small air sacs known as alveoli. Pneumonia may present with opacities.
            covid-19: A contagious disease caused by a virus. Ground-glass opacities, consolidation, thickening, pleural effusions commonly appear in infection.
            Choose the option letter that describes the image best.
            [/INST] Understood.</s>[INST] """

            prompts = [
                idx[0], 
                """Question: What illness does the patient have? 
                A. normal
                B. pneumonia
                C. covid-19 
                Answer: """
            ] 
            #prompts = [idx[0], "Question: Is the chest x-ray image normal, non-COVID illness, or COVID-19? Answer: "] 
            model_ans = model.classification_for_eval(prompts, covid_classes, add_special_tokens=False) # top_p=0.9, temperature=0.5
            model_ans = model_ans.translate(str.maketrans('', '', string.punctuation)) # remove punctuation

            # print(f"Model Prediction: {model_ans}, Ground Truth: {ans}")

            if model_ans.lower() == ans.lower():
                right_answers += 1

            total_answers += 1    

    print(f"COVID Classification Accuracy: {right_answers} / {total_answers} = {(right_answers / total_answers)*100}")


for dname in eval_datasets:
    assert dname in ["rsna", "covid", "vqarad"]
    if dname == "covid":
        COVID_evaluation()
    elif dname == "rsna":
        RSNA_evaluation()
    elif dname == "vqarad":
        VQARAD_evaluation()