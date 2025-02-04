from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
import torch
import pandas as pd
import os
import argparse
from tqdm import tqdm
import json

parser = argparse.ArgumentParser()
parser.add_argument('--epi', type=str, default='LPRRSGAAGA')
parser.add_argument('--n_seq', type=int, default=1000)
parser.add_argument('--device', type=int, default=2)
parser.add_argument('--k_shots', type=int, default=1)

parser.add_argument('--model_data_epis_n', type=str, default='contains_greaterFreq_100_epis')
parser.add_argument('--model_data_mode', type=str, default='out_of_sample')
parser.add_argument('--model_name', type=str, default='rita_m')
args = parser.parse_args()

## Load TCR embedding model -- catELMo
model_dir = Path('/path/of/catELMo/checkpoints')          
weights = model_dir/'weights.hdf5'
options = model_dir/'options.json'
embedder  = ElmoEmbedder(options,weights,cuda_device=args.device) # cuda_device=-1 for CPU

# Load epitope embeddings -- from blosum62, size 528
with open('/path/to/epi_blosum62_embeddings_528.json', 'r') as f:
    epi_embedding_dict = json.load(f)
    
        
def ELMo_embeds(x):
    return torch.tensor(embedder.embed_sentence(list(x))).sum(dim=0).mean(dim=0).tolist()


dat1 = pd.read_csv(f'../designed_TCRs/{args.epi}_{args.k_shots}_shots.csv')
dat1['tcr_embeds'] = None
dat1['epi_embeds'] = None

# elmo_embedding = ELMo_embeds(dat1['epi'][0]) # only need to be embeded once.
# Use tqdm to create a progress bar
for index, row in tqdm(dat1.iterrows(), total=dat1.shape[0]):
    dat1.at[index, 'tcr_embeds'] = ELMo_embeds(row['tcr'])
    dat1.at[index, 'epi_embeds'] = epi_embedding_dict[row['epi']]

    
dat1.to_pickle(f"../embeddings/{args.epi}_{args.k_shots}_shots.pkl")
