#!/bin/bash

conda activate gpu-venv-study-hrp

export TFHUB_CACHE_DIR=./models-tfhub

mkdir -p $TFHUB_CACHE_DIR

declare -a MODELS=(
    universal-sentence-encoder-multilingual/3
)

for MODEL in ${MODELS[*]} ; do
    python -c "import tensorflow_hub; model = tensorflow_hub.load('https://tfhub.dev/google/${MODEL}')"
done
