from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Blast.Applications import NcbimakeblastdbCommandline,NcbiblastpCommandline
import pandas as pd
import os
import multiprocessing

os.makedirs("dbs", exist_ok=True)
os.makedirs("fasta_files", exist_ok=True)
os.makedirs("results_temp", exist_ok=True)

gen_tcr_dict = {}
natural_tcr_dict = {}

def write_fasta(epitope_name,file_name):

    natural_tcrs_for_current_epitope = natural_tcr.loc[(natural_tcr["epi"]==epitope_name) & (natural_tcr["binding"]==1)].reset_index().drop(columns=['index'])
    natural_tcr_sequences = natural_tcrs_for_current_epitope["tcr"].values

    generated_tcrs_for_current_epitope = pd.read_csv(f"{original_path}/{file_name}.csv")
            
    generated_tcr_sequences  = list(set(generated_tcrs_for_current_epitope["tcr"].values))
    records = []
    
    for i, sequence in enumerate(generated_tcr_sequences, 1):
        if "\n" in sequence or "\r" in sequence:
            continue
        gen_tcr_dict[f"Generated_TCR_{i}"] = sequence
        record = SeqRecord(Seq(sequence), id=f"Generated_TCR_{i}", description="")
        records.append(record)
    
    with open(f"fasta_files/{file_name}_generated.fasta", "w") as fasta_file:
        SeqIO.write(records, fasta_file, "fasta")
        
    generated_seqs = SeqIO.to_dict(SeqIO.parse(f"fasta_files/{file_name}_generated.fasta", "fasta"))
    
    records = []

    for i, sequence in enumerate(natural_tcr_sequences, 1):
        natural_tcr_dict[f"Natural_TCR_{i}"] = sequence
        record = SeqRecord(Seq(sequence), id=f"Natural_TCR_{i}", description="")
        records.append(record)
    
    with open(f"fasta_files/{file_name}_natural.fasta", "w") as fasta_file:
        SeqIO.write(records, fasta_file, "fasta")
    
    return generated_seqs

def create_natural_tcr_db(file_name):
    makeblastdb_cline = NcbimakeblastdbCommandline(
        input_file=f"fasta_files/{file_name}_natural.fasta",  
        dbtype="prot",                         
        out=f"dbs/{file_name}_natural_db"                        
    )

    stdout, stderr = makeblastdb_cline()

def calculate_scores_using_blastp(file_name):
    blastp_cline = NcbiblastpCommandline(
        query=f"fasta_files/{file_name}_generated.fasta",  
        evalue=0.001,     
        db=f"dbs/{file_name}_natural_db" ,                  
        out=f"results_temp/results_{file_name}.txt",           
        outfmt="6 qseqid qseq sseq bitscore"    
    )

    stdout, stderr = blastp_cline()

def format_file_and_generate_summary(epitope_name,generated_seqs,file_name):
    columns = ["qseqid","qseq","sseq","bitscore"]
    df = pd.read_csv(f"results_temp/results_{file_name}.txt", sep="\t", names=columns)

    df["subject_seq_full"] = df["qseqid"].map(lambda x: str(generated_seqs[x].seq) if x in generated_seqs else "")

    agg_df = (
    df.groupby(['subject_seq_full'])['bitscore']
    .agg(['max', 'min', 'mean'])
    .reset_index()
    .rename(columns={'max': 'max_bitscore', 'min': 'min_bitscore', 'mean': 'avg_bitscore'})
    )[['subject_seq_full','max_bitscore','min_bitscore','avg_bitscore']]

    agg_df["epitope"] = epitope_name
    

    agg_df.columns = ['generated_tcr','max_bitscore','min_bitscore','avg_bitscore','epitope']

    agg_df[['epitope','generated_tcr','max_bitscore','min_bitscore','avg_bitscore']].to_csv(f"{os.path.join(new_base, relative_subdirs)}/{file_name}.csv")

def generate_bit_score_pipeline(file_name):
    if ".ipynb" in file_name:
        return
    
    # Comment if the results need not be rewritten
    if os.path.isfile(f"{os.path.join(new_base, relative_subdirs)}/{file_name}"):
        return
    
    epitope_name = file_name.split("_")[0]

    try:
        generated_seqs = write_fasta(epitope_name,file_name.split(".")[0])
        create_natural_tcr_db(file_name.split(".")[0])
        calculate_scores_using_blastp(file_name.split(".")[0])
        format_file_and_generate_summary(epitope_name,generated_seqs,file_name.split(".")[0])
    
    except:
        print(f"Error occured for epitope {epitope_name}")

if __name__ == '__main__':

    natural_tcr = pd.read_csv("TCREpitopePairs.csv")

    #Use these samples if gen5 or gen10
    #["fsp-fake",'fsp-healthy','fsp-oracle','scp-chain','scp-random','scp-select']

    # Use these samples if gen0 (change based on folder name)
    #"in_sample/temper_0.4","out_of_sample/temper_0.4"

    for sample in ["in_sample/temper_0.4","out_of_sample/temper_0.4"]:
        original_path = f"/home/sprabh35/datadisk/tcr_evaluation/our_model_generation/gen0/{sample}/designed_TCRs"

        #Path where we store the results
        new_base = "/home/sprabh35/datadisk/tcr_evaluation/tcr_evaluations/evaluations_bitscore_tcrgen_new"

        #Code to extract only required path to create new directories
        relative_subdirs = os.path.relpath(os.path.dirname(original_path), "/home/sprabh35/datadisk/tcr_evaluation/our_model_generation")

        os.makedirs(os.path.join(new_base, relative_subdirs), exist_ok=True)

        files = os.listdir(original_path)

        with multiprocessing.Pool(processes=20) as pool:
            results = pool.map(generate_bit_score_pipeline, files)

        print(f"Processing finished for sample {sample}")