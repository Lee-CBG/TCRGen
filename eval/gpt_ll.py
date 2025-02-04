import argparse
import torch
from transformers import pipeline
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--epi', type=str, default='LPRRSGAAGA')
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--k_shots', type=int, default=1)
parser.add_argument('--model_data_epis_n', type=str, default='contains_greaterFreq_100_epis')
parser.add_argument('--model_data_mode', type=str, default='out_of_sample')
parser.add_argument('--model_name', type=str, default='rita_m')
args = parser.parse_args()



def compute_log_likelihood(generated_tcr, model, tokenizer, device):
    model.to(device)
    encoded_input = tokenizer(generated_tcr, return_tensors="pt", padding=True, truncation=True)

    # Move the input tensors to the specified device
    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

    # Feed the encoded input to the GPT model
    with torch.no_grad():
        outputs = model(**encoded_input)

    # Retrieve the logits (raw outputs) from the GPT model
    logits = outputs.logits

    # Initialize variables to store total log likelihood and sequence length
    total_log_likelihood = 0.0
    sequence_length = len(generated_tcr)

    # Iterate over each amino acid position in the generated TCR sequence
    for i, amino_acid in enumerate(generated_tcr):
        log_likelihood = logits[0, i, tokenizer.convert_tokens_to_ids(amino_acid)]
        total_log_likelihood += log_likelihood

    # Compute the average log likelihood
    average_log_likelihood = total_log_likelihood / sequence_length

    return total_log_likelihood.cpu().numpy()#, average_log_likelihood


# Read the generated TCRs
designed_TCRs = pd.read_csv(f'../designed_TCRs/{args.epi}_{args.k_shots}_tcrs_prompting.csv')

if 'gpt_ll' in designed_TCRs.columns:
    print(f"'gpt_ll' column already exists in {args.epi}_{args.k_shots}_tcrs_prompting.csv. Skipping computation.")
else:
    print(f"'gpt_ll' column missing in {args.epi}_{args.k_shots}_tcrs_prompting.csv. Recomputing...")

    MODEL_DIR = "/mnt/disk07/user/pzhang84/generativeTCR/finetuneRITA_2/models_ft_1/rita_m/checkpoint-6400"
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("/mnt/disk07/user/pzhang84/generativeTCR/finetuneRITA_2/models_RITA_add_tokens/RITA_m")

    # Add special tokens (if not already added during training)
    special_tokens_dict = {'eos_token': '<EOS>', 'pad_token': '<PAD>', 'additional_special_tokens': ['$','<tcr>','<eotcr>','<epi>','<epepi>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer)) 
    tokenizer.pad_token = tokenizer.eos_token

    # Check if the specified GPU is available, otherwise fall back to CPU
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")



    log_likelihoods = [0] * len(designed_TCRs)
    for i, designed_tcr in enumerate(designed_TCRs['tcr']):
        log_likelihoods[i] = compute_log_likelihood(designed_tcr, model, tokenizer, device)

    # Add the max score as a new column in the DataFrame
    designed_TCRs['gpt_ll'] = log_likelihoods
    designed_TCRs.to_csv(f'../designed_TCRs/{args.epi}_{args.k_shots}_tcrs_prompting.csv', index=False)