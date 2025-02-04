import os
import warnings
import random
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description="TCR Generation Script")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID to use")
parser.add_argument("--gen_model", type=str, default='10', help="10 or 5")
args = parser.parse_args()


# Suppress warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
warnings.filterwarnings('ignore', category=FutureWarning)

seed = 42 
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


# Constants
MODEL_DIR = f"./../models/SFT/rita_m/{args.gen_model}_shots"
TOKENIZER_DIR = "./../models/pLMs/RITA_m"
RESULTS_DIR = f"./results/gen{args.gen_model}/scp-chain/designed_TCRs"
EPITOPES_FILE = './../data/epitopes.txt'


#### Load model checkpoints and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
special_tokens_dict = {'eos_token': '<EOS>', 'pad_token': '<PAD>'}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))  
tokenizer.pad_token = tokenizer.eos_token

#### load model to cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

#### Paramters
if args.gen_model == '10':
    max_length_param = 64*5
    k_shots = 9  
elif args.gen_model == '5':
    max_length_param = 64*3
    k_shots = 4
    
do_sample_param = True
top_k_param = 8
repetition_penalty_param = 1.0
eos_token_id_param = 2
batch_size = 300 
num_batches = 1  
temperature = 0.4


# load epitopes list that we are interested to generate TCRs for
with open(EPITOPES_FILE, 'r') as file:
    epitopes = [line.strip() for line in file if line.strip()]


#### TCR Generation with SCP random
for EPITOPE in epitopes:
    EPITOPE_PROMPT = EPITOPE + '$'
    
    outputs = []
    for _ in tqdm(range(num_batches)):
        output = text_generator(EPITOPE_PROMPT, max_length=max_length_param, do_sample=do_sample_param, 
                                   top_k=top_k_param, repetition_penalty=repetition_penalty_param,
                                   num_return_sequences=batch_size, eos_token_id=eos_token_id_param,
                                   temperature=0.4)
        outputs.extend(output)



    for shot in range(0, k_shots + 1):
        print(f'Saving Epitope-TCR pairs for shot {shot} into a csv file...')
        output_file = os.path.join(RESULTS_DIR, f"{EPITOPE}_{shot}_tcrs_prompting.csv")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epi", "tcr"])
    
            for output in outputs:
                # Replace spaces and split by '$'
                split_text = output["generated_text"].replace(' ', '').split('$')
    
                try:
                    # Ensure valid indexing for the current shot
                    epi = split_text[0]  # Epitope is always the first part
                    tcr = split_text[shot + 1]  # TCR corresponding to the current shot
                except IndexError:
                    tcr = "AA"  # Default to "AA" if no TCR is found
    
                # Ensure TCR has at least two characters
                if len(tcr) <= 1:
                    tcr = "AA"
                
                # Write the epitope and TCR for the current shot
                writer.writerow([epi, tcr])

