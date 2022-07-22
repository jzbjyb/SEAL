#!/usr/bin/env bash
#SBATCH --job-name=seal
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=30:00
#SBATCH --partition=learnlab
#SBATCH --mem=64GB
#SBATCH --constraint=volta32gb
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err

# env
source env.sh

# arguments
model_class=bart
model=data/SEAL-checkpoint+index.KILT/SEAL.KILT.pt
index=data/SEAL-checkpoint+index.KILT/KILT.fm_index
input=output/sealkilt_fruit_preceding.json
output=output/sealkilt_fruit_preceding.bart_ref_first3.json
context_type=ctxs

python -m seal.generate \
    --input ${input} \
    --output ${output} \
    --model_class ${model_class} \
    --context_type ${context_type} \
    --checkpoint ${model} \
    --fm_index ${index} \
    --device cuda:1 \
    --batch_size 20 \
    --beam 5
