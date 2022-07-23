#!/usr/bin/env bash

model=$1
index=data/SEAL-checkpoint+index.NQ/NQ.fm_index
data=/checkpoint/zhengbao/exp/side/data/edit_eval/fruit_gold/test-dpr-preceding-inst0.json
output=$2
jobs=$3

TOKENIZERS_PARALLELISM=false python -m seal.search \
    --topics_format dpr --topics ${data} \
    --output_format dpr --output ${output} \
    --checkpoint ${model} \
    --fm_index ${index} \
    --jobs ${jobs} \
    --progress \
    --device cuda:0 \
    --batch_size 20 \
    --beam 15
