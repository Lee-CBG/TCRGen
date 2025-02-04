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
RESULTS_DIR = f"./results/gen{args.gen_model}/fsp-healthy/designed_TCRs"
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


# Loop over each epitope
for EPITOPE in epitopes:
    EPITOPE_PROMPT_BASE = EPITOPE + '$'
    
    # Load the corresponding TCR data for the current epitope
    df_tcrs = pd.read_csv(f'./../data/healthy_tcrs.csv')

    
    # Convert the filtered TCRs to a list
    selected_tcrs = df_tcrs['tcr'].to_list()

    # Loop for each shot level
    for shot in range(1, k_shots):
        outputs = []
        
        # Build the prompt for the current number of shots
        prompt = EPITOPE_PROMPT_BASE
        for i in range(shot):
            if i < len(selected_tcrs):
                prompt += random.choice(selected_tcrs) + '$'
            else:
                break  # Stop if there aren't enough selected TCRs
        
        # Generate outputs for the current prompt
        for _ in tqdm(range(num_batches)):
            output = text_generator(prompt, max_length=max_length_param, do_sample=do_sample_param,
                                    top_k=top_k_param, repetition_penalty=repetition_penalty_param,
                                    num_return_sequences=batch_size, eos_token_id=eos_token_id_param, temperature=temperature)
            outputs.extend(output)

        
        # Save generated sequences
        print(f'Saving Epitope-TCR pairs for shot {shot} into a csv file...')
        output_file = os.path.join(RESULTS_DIR, f"{EPITOPE}_{shot}_tcrs_prompting.csv")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epi", "tcr"])

            for output in outputs:
                split_text = output["generated_text"].replace(' ', '').split('$')
                
                try:
                    # Get the epitope and the (k_shots+1)th TCR sequence
                    epi = split_text[0]
                    tcr = split_text[shot+1]
                except IndexError:
                    tcr = "AA"  # Default to "AA" if no TCR is found

                # Ensure tcr has at least two characters
                if len(tcr) <= 1:
                    tcr = "AA" 

                writer.writerow([epi, tcr])
                