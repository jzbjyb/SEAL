#!/usr/bin/env bash

input=$1
output=$2

python scripts/training/make_supervised_dpr_dataset.py ${input} ${output} --target title --mark_target --mark_silver --n_samples 3 --mode a
python scripts/training/make_supervised_dpr_dataset.py ${input} ${output} --target span --mark_target --mark_silver --n_samples 10 --mode a
