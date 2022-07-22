#!/usr/bin/env bash
#SBATCH --job-name=seal
#SBATCH --cpus-per-task=80
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --partition=learnlab
#SBATCH --mem=64GB
#SBATCH --constraint=volta32gb
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err

# env
source env.sh

./scripts/training/training_fairseq.sh "$@"
