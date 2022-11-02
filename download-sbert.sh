#!/bin/bash

conda activate gpu-venv-study-hrp

export SENTENCE_TRANSFORMERS_HOME=./models-sbert

mkdir -p $SENTENCE_TRANSFORMERS_HOME

declare -a MODELS=(
    paraphrase-multilingual-mpnet-base-v2 
    paraphrase-multilingual-MiniLM-L12-v2 
    distiluse-base-multilingual-cased-v2 
    sentence-transformers/LaBSE
)

# all-mpnet-base-v2 
# all-distilroberta-v1 
# all-MiniLM-L12-v2 
# all-MiniLM-L6-v2


for MODEL in ${MODELS[*]} ; do
    python -c "import sentence_transformers; sentence_transformers.SentenceTransformer('${MODEL}')"
done
