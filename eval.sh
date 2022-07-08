#!/usr/bin/env bash
#SBATCH --cpus-per-task=10
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=1800
#SBATCH --partition=learnlab
#SBATCH --mem=470GB
#SBATCH --signal=USR1@140
#SBATCH --constraint=volta32gb
#SBATCH --job-name=per_8x8_gold_ref_no_ref
#SBATCH -o /private/home/schick/logs/plan_edit_repeat/sample-%j.out
#SBATCH -e /private/home/schick/logs/plan_edit_repeat/sample-%j.err

model=data/SEAL-checkpoint+index.KILT/SEAL.KILT.pt
index=data/SEAL-checkpoint+index.KILT/KILT.fm_index
#data=data/nq/biencoder-nq-dev.json
data=/checkpoint/zhengbao/exp/side/data/edit_eval/fruit/test-dpr-preceding-inst0.json
output=output/sealkilt_fruit_preceding.json

TOKENIZERS_PARALLELISM=false python -m seal.search \
    --topics_format dpr --topics ${data} \
    --output_format dpr --output ${output} \
    --checkpoint ${model} \
    --fm_index ${index} \
    --jobs 64 --progress --device cuda:1 --batch_size 20 \
    --beam 15
