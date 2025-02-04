import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epi', type=str, default='LPRRSGAAGA')
parser.add_argument('--n_seq', type=int, default=1000)
parser.add_argument('--k_shots', type=int, default=1)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--model_data_epis_n', type=str, default='contains_greaterFreq_100_epis')
parser.add_argument('--model_data_mode', type=str, default='out_of_sample')
parser.add_argument('--model_name', type=str, default='rita_m')
args = parser.parse_args()


def TCRMatch(seq1, seq2):
    # Initialize variables to store the results
    similarity_score = 0.0
    normalization_factor = 0.0

    # Determine the minimum length between seq1 and seq2
    min_length = min(len(seq1), len(seq2))

    # Calculate similarity for each k-mer size (k = 1 to min_length)
    for k in range(1, min_length + 1):
        similarity_kmer = 0.0

        for i in range(len(seq1) - k + 1):
            kmer1 = seq1[i:i+k]
            for j in range(len(seq2) - k + 1):
                kmer2 = seq2[j:j+k]
                kmer_similarity = 1.0  # Initialize k-mer similarity to 1.0

                # Calculate k-mer similarity using BLOSUM62 values
                for p in range(k):
                    amino_acid1 = amino_map[kmer1[p]]
                    amino_acid2 = amino_map[kmer2[p]]
                    kmer_similarity *= blosum62[amino_acid1][amino_acid2]

                similarity_kmer += kmer_similarity

        # Add the k-mer similarity to the overall similarity score
        similarity_score += similarity_kmer
    return similarity_score


def TCRMatchNorm(seq1, seq2):
    normalization_factor = np.sqrt(TCRMatch(seq1, seq1)* TCRMatch(seq2, seq2))
    similarity = TCRMatch(seq1, seq2) / normalization_factor
#     print("Similarity Score:", similarity)
    return similarity


amino_map = {'@':24, '*': 23, 'A': 0, 'C': 4, 'B': 20,
             'E': 6, 'D': 3, 'G': 7, 'F': 13, 'I': 9, 'H': 8,
             'K': 11, 'M': 12, 'L': 10, 'N': 2, 'Q': 5, 'P': 14,
             'S': 15, 'R': 1, 'T': 16, 'W': 17, 'V': 19, 'Y': 18,
             'X': 22, 'Z': 21}

# the matrix that used in TCRMatch
blosum62 = [
    [0.0215, 0.0023, 0.0019, 0.0022, 0.0016, 0.0019, 0.003,
     0.0058, 0.0011, 0.0032, 0.0044, 0.0033, 0.0013, 0.0016,
     0.0022, 0.0063, 0.0037, 0.0004, 0.0013, 0.0051],
    [0.0023, 0.0178, 0.002,  0.0016, 0.0004, 0.0025, 0.0027,
     0.0017, 0.0012, 0.0012, 0.0024, 0.0062, 0.0008, 0.0009,
     0.001,  0.0023, 0.0018, 0.0003, 0.0009, 0.0016],
    [0.0019, 0.002,  0.0141, 0.0037, 0.0004, 0.0015, 0.0022,
     0.0029, 0.0014, 0.001,  0.0014, 0.0024, 0.0005, 0.0008,
     0.0009, 0.0031, 0.0022, 0.0002, 0.0007, 0.0012],
    [0.0022, 0.0016, 0.0037, 0.0213, 0.0004, 0.0016, 0.0049,
     0.0025, 0.001,  0.0012, 0.0015, 0.0024, 0.0005, 0.0008,
     0.0012, 0.0028, 0.0019, 0.0002, 0.0006, 0.0013],
    [0.0016, 0.0004, 0.0004, 0.0004, 0.0119, 0.0003, 0.0004,
     0.0008, 0.0002, 0.0011, 0.0016, 0.0005, 0.0004, 0.0005,
     0.0004, 0.001,  0.0009, 0.0001, 0.0003, 0.0014],
    [0.0019, 0.0025, 0.0015, 0.0016, 0.0003, 0.0073, 0.0035,
     0.0014, 0.001,  0.0009, 0.0016, 0.0031, 0.0007, 0.0005,
     0.0008, 0.0019, 0.0014, 0.0002, 0.0007, 0.0012],
    [0.003,  0.0027, 0.0022, 0.0049, 0.0004, 0.0035, 0.0161,
     0.0019, 0.0014, 0.0012, 0.002,  0.0041, 0.0007, 0.0009,
     0.0014, 0.003,  0.002,  0.0003, 0.0009, 0.0017],
    [0.0058, 0.0017, 0.0029, 0.0025, 0.0008, 0.0014, 0.0019,
     0.0378, 0.001,  0.0014, 0.0021, 0.0025, 0.0007, 0.0012,
     0.0014, 0.0038, 0.0022, 0.0004, 0.0008, 0.0018],
    [0.0011, 0.0012, 0.0014, 0.001,  0.0002, 0.001,  0.0014,
     0.001,  0.0093, 0.0006, 0.001,  0.0012, 0.0004, 0.0008,
     0.0005, 0.0011, 0.0007, 0.0002, 0.0015, 0.0006],
    [0.0032, 0.0012, 0.001,  0.0012, 0.0011, 0.0009, 0.0012,
     0.0014, 0.0006, 0.0184, 0.0114, 0.0016, 0.0025, 0.003,
     0.001,  0.0017, 0.0027, 0.0004, 0.0014, 0.012],
    [0.0044, 0.0024, 0.0014, 0.0015, 0.0016, 0.0016, 0.002,
     0.0021, 0.001,  0.0114, 0.0371, 0.0025, 0.0049, 0.0054,
     0.0014, 0.0024, 0.0033, 0.0007, 0.0022, 0.0095],
    [0.0033, 0.0062, 0.0024, 0.0024, 0.0005, 0.0031, 0.0041,
     0.0025, 0.0012, 0.0016, 0.0025, 0.0161, 0.0009, 0.0009,
     0.0016, 0.0031, 0.0023, 0.0003, 0.001,  0.0019],
    [0.0013, 0.0008, 0.0005, 0.0005, 0.0004, 0.0007, 0.0007,
     0.0007, 0.0004, 0.0025, 0.0049, 0.0009, 0.004,  0.0012,
     0.0004, 0.0009, 0.001,  0.0002, 0.0006, 0.0023],
    [0.0016, 0.0009, 0.0008, 0.0008, 0.0005, 0.0005, 0.0009,
     0.0012, 0.0008, 0.003,  0.0054, 0.0009, 0.0012, 0.0183,
     0.0005, 0.0012, 0.0012, 0.0008, 0.0042, 0.0026],
    [0.0022, 0.001,  0.0009, 0.0012, 0.0004, 0.0008, 0.0014,
     0.0014, 0.0005, 0.001,  0.0014, 0.0016, 0.0004, 0.0005,
     0.0191, 0.0017, 0.0014, 0.0001, 0.0005, 0.0012],
    [0.0063, 0.0023, 0.0031, 0.0028, 0.001,  0.0019, 0.003,
     0.0038, 0.0011, 0.0017, 0.0024, 0.0031, 0.0009, 0.0012,
     0.0017, 0.0126, 0.0047, 0.0003, 0.001,  0.0024],
    [0.0037, 0.0018, 0.0022, 0.0019, 0.0009, 0.0014, 0.002,
     0.0022, 0.0007, 0.0027, 0.0033, 0.0023, 0.001,  0.0012,
     0.0014, 0.0047, 0.0125, 0.0003, 0.0009, 0.0036],
    [0.0004, 0.0003, 0.0002, 0.0002, 0.0001, 0.0002, 0.0003,
     0.0004, 0.0002, 0.0004, 0.0007, 0.0003, 0.0002, 0.0008,
     0.0001, 0.0003, 0.0003, 0.0065, 0.0009, 0.0004],
    [0.0013, 0.0009, 0.0007, 0.0006, 0.0003, 0.0007, 0.0009,
     0.0008, 0.0015, 0.0014, 0.0022, 0.001,  0.0006, 0.0042,
     0.0005, 0.001,  0.0009, 0.0009, 0.0102, 0.0015],
    [0.0051, 0.0016, 0.0012, 0.0013, 0.0014, 0.0012, 0.0017,
     0.0018, 0.0006, 0.012,  0.0095, 0.0019, 0.0023, 0.0026,
     0.0012, 0.0024, 0.0036, 0.0004, 0.0015, 0.0196]]



random.seed(42)  

# Read the generated TCRs
designed_TCRs = pd.read_csv(f'../designed_TCRs/{args.epi}_{args.k_shots}_tcrs_prompting.csv')

# Read the real TCRs for the specified epitope
df = pd.read_csv('/mnt/disk07/user/pzhang84/generativeTCR/finetuneRITA_2/data/epitope_2_tcrs/combined_dataset_repTCRs.csv')
df = df[:150008]  # Only use positively bind pairs
real_TCRs = df[df['epi'] == args.epi].reset_index(drop=True)

# Preselect 50 real TCRs for each generated TCR
selected_real_tcrs = {}
for designed_tcr in designed_TCRs['tcr']:
#     selected_real_tcrs[designed_tcr] = random.choices(real_TCRs['tcr'].tolist(), k=50)
    selected_real_tcrs[designed_tcr] = real_TCRs['tcr'].tolist()

# Predefine the size of the scores array
num_generated_tcrs = len(designed_TCRs)
num_preselected_tcrs = len(real_TCRs) # 50
scores = np.zeros((num_generated_tcrs, num_preselected_tcrs))


# Compute match scores for each pair of generated and preselected real TCRs
for i, designed_tcr in enumerate(tqdm(designed_TCRs['tcr'], desc="Computing match scores")):
    for j, real_tcr in enumerate(selected_real_tcrs[designed_tcr]):
        scores[i, j] = TCRMatchNorm(designed_tcr, real_tcr)


# Select the maximum score for each generated TCR
max_scores = np.max(scores, axis=1)
designed_TCRs['tcr_match'] = max_scores

# Save the updated DataFrame to a CSV file
designed_TCRs.to_csv(f'../designed_TCRs/{args.epi}_{args.k_shots}_tcrs_prompting.csv', index=False)