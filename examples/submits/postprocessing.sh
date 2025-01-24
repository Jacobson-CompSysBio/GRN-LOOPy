#!/bin/bash

# Filepath to the processed data
INPUT_FILEPATH="../data/processed_data_set.tsv"
REP_MAP_PATH="../data/examples/data/raw_data_ge1e-05variance_nonrep_to_rep_map.tsv"
PROJECT_DIRECTORY_PATH="../"

iRFLOOP_PATH="../../src/postprocess.py" 

# cd $PROJECT_DIRECTORY_PATH
SECONDS=0
python $iRFLOOP_PATH \
        --infile $INPUT_FILEPATH \
        --delim "\t" \
        --threshold 0.01 \
        --rep_map_path $REP_MAP_PATH \
        --verbose
echo $SECONDS elapsed

