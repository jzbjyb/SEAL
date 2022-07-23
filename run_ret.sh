#!/usr/bin/env bash
#SBATCH --job-name=seal
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=10:00:00
#SBATCH --partition=learnlab
#SBATCH --mem=256GB
#SBATCH --constraint=volta32gb
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err

# env
source env.sh

srun ret.sh "$@"
