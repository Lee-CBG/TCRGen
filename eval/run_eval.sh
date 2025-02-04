#!/bin/sh
DEVICE=2

# Loop over K values from 0 to 4
for K in 0 1 2 3 4
do
    echo "Starting processing for K = $K..."

    while IFS= read -r EPI
    do
        echo "Processing K = $K..."

        echo "log_likelihood..."
        source activate GPT-LL
        python gpt_ll.py --k_shots $K --epi $EPI --device $DEVICE #2>/dev/null
        conda deactivate

        echo "Embedding TCR sequences..."
        source activate torch14_conda
        python embedding.py --k_shots $K --epi $EPI --device $DEVICE #2>/dev/null  # Redirect standard error to /dev/null
        conda deactivate

        echo "Predicting binding affinity scores..."
        source activate tf26
        python bap.py --k_shots $K --epi $EPI #2>/dev/null  # Redirect standard error to /dev/null

        echo "Computing TCRMatch Scores..."
        python tcr_match.py --k_shots $K --epi $EPI #2>/dev/null  # Redirect standard error to /dev/null
        conda deactivate

    done < epitope.txt
done
