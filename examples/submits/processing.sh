#!/bin/bash

# Filepath to the preprocessing data
INPUT_FILEPATH="../data/preprocessed.tsv"
PROJECT_DIRECTORY_PATH="../"
OUTPUT_FILE_PATH="${PROJECT_DIRECTORY_PATH}/data/processed_data_set.tsv"

iRFLOOP_PATH="../../src/process.py" 

# cd $PROJECT_DIRECTORY_PATH
SECONDS=0
python $iRFLOOP_PATH --mpi\
        --infile $INPUT_FILEPATH \
        --header_row_idx 0 \
        --delim "\t" \
        --device cpu \
        --model_name lgbm \
        --objective regression \
        --n_estimators 100 \
        --num_leaves 20 \
        --max_depth 5 \
        --random_state 42 \
        --calc_permutation_importance \
        --calc_permutation_score \
        --n_permutations 100 \
        --outfile $OUTPUT_FILE_PATH \
        --n_processes 10 \
        --verbose 1
echo $SECONDS elapsed

