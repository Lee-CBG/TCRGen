import pandas as pd
from Bio.Align import substitution_matrices
matrix = substitution_matrices.load('BLOSUM62')
from Bio.Align import PairwiseAligner
import numpy as np

aligner = PairwiseAligner()
aligner.mode = "local"
aligner.substitution_matrix = matrix

import os
import multiprocessing

natural_tcr = pd.read_csv("TCREpitopePairs.csv")


def evaluate_tcrs(file_name):
    epitope_name = file_name.split("_")[0]

    # Comment if results need not be rewritten
    if os.path.isfile(f"{os.path.join(new_base, relative_subdirs)}/local/{file_name}"):
        return

    print(f"-------------------Processing started for epitope {file_name}-------------------")

    try:
        natural_tcrs_for_current_epitope = natural_tcr.loc[(natural_tcr["epi"]==epitope_name) & (natural_tcr["binding"]==1)].reset_index().drop(columns=['index'])
        
        generated_tcrs_for_current_epitope = pd.read_csv(f"{original_path}/{file_name}")
        
        natural_tcr_arr = natural_tcrs_for_current_epitope["tcr"].values
        generated_tcr_arr  = list(set(generated_tcrs_for_current_epitope["tcr"].values))

        score_map = {tcr: [] for tcr in generated_tcr_arr}
        
        for nat in natural_tcr_arr:
            for gen in generated_tcr_arr:
                try:
                    alignments = aligner.align(nat.upper(), gen.upper())
                    if len(alignments):
                        res = alignments[0]
                        score_map[gen].append(res.score)
                except Exception as e:
                    print(f"Error {e}")
        
        output_rows = []
        for tcr in generated_tcr_arr:
            max_score = max(score_map[tcr]) if score_map[tcr] else 0
            min_score = min(score_map[tcr]) if score_map[tcr] else 0
            avg_score = round(np.mean(score_map[tcr]) if score_map[tcr] else 0,4)
            output_rows.append({'epitope': epitope_name, 'binding_tcr': tcr, 'max_blosum62_score': max_score,'min_blosum62_score':min_score,'avg_blosum62_score':avg_score})
                    
        pd.DataFrame(output_rows,columns=["epitope","binding_tcr","max_blosum62_score","min_blosum62_score","avg_blosum62_score"]).to_csv(f"{os.path.join(new_base, relative_subdirs)}/local/{file_name}")

    except Exception as e:
        print(f"Error occured for epitope {file_name}")

    print(f"-------------------Processing finished for epitope {file_name}-------------------")


if __name__ == '__main__':

    # Sample list if gen5/gen10 is used
    #"fsp-oracle","fsp-fake","fsp-healthy","scp-chain","scp-random","scp-select"

    # Sample list if gen0 is usedd
    #"in_sample/temper_0.4","out_of_sample/temper_0.4"

    for sample in ["fsp-oracle","fsp-fake","fsp-healthy","scp-chain","scp-random","scp-select"]:

        original_path = f"/home/sprabh35/datadisk/tcr_evaluation/our_model_generation/gen10/{sample}/designed_TCRs"
       
        #Path where we store the results
        new_base = "/home/sprabh35/datadisk/tcr_evaluation/tcr_evaluations/evaluations_blosum_tcrgen_new/"

        #Code to extract only required path to create new directories
        relative_subdirs = os.path.relpath(os.path.dirname(original_path), "/home/sprabh35/datadisk/tcr_evaluation/our_model_generation")

        os.makedirs(f"{os.path.join(new_base, relative_subdirs)}/local", exist_ok=True)

        files = [f for f in os.listdir(original_path) if f != '.ipynb_checkpoints']

        with multiprocessing.Pool(processes=20) as pool:
            results = pool.map(evaluate_tcrs, files)

        
        print(f"Processing finished for {sample}")