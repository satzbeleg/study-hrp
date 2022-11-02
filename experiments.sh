#!/bin/bash

conda activate gpu-venv-study-hrp

export SENTENCE_TRANSFORMERS_HOME=./models-sbert
export TFHUB_CACHE_DIR=./models-tfhub

declare -a MODELS=(
    paraphrase-multilingual-mpnet-base-v2 
    paraphrase-multilingual-MiniLM-L12-v2 
    distiluse-base-multilingual-cased-v2 
    sentence-transformers/LaBSE
    laser-de
    laser-en
    m-use
)

# for output-type=hrp
declare -a NUMBOOLFEATS=(256 384 512 768 1024 1536 2048)

# random seeds
declare -a SEEDS=(23 24 25 26 27 28 29 30 31 32)


for MODEL in ${MODELS[*]} ; do
    # run sigmoid
    python3 script.py --model=$MODEL --output-type=sigmoid

    # run original sentence embeddings
    python3 script.py --model=$MODEL --output-type=float

    # run hrp
    for NUMBOOLFEAT in ${NUMBOOLFEATS[*]} ; do
        for SEED in ${SEEDS[*]} ; do
            python3 script.py \
                --model=$MODEL \
                --num-bool-features=$NFEATS \
                --random-state=$SEED \
                --output-type=hrp
        done
    done
done
