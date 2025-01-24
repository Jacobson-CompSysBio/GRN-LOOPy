#!/bin/bash

# Filepath to the preprocessing data
INPUT_FILEPATH="../data/raw_data.tsv"
PROJECT_DIRECTORY_PATH="../"
OUTPUT_FILE_PATH="${PROJECT_DIRECTORY_PATH}/data/preprocessed.tsv"

iRFLOOP_PATH="../../src/preprocess.py" 

# cd $PROJECT_DIRECTORY_PATH

SECONDS=0
python $iRFLOOP_PATH \
        --infile $INPUT_FILEPATH \
        --remove_low_variance \
        --cv_thresh 0.00001 \
        --remove_high_corr \
        --corr_thresh 0.95 \
        --outfile $OUTPUT_FILE_PATH

echo $SECONDS elapsed