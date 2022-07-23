# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

model=$1  # which model to start from 
if [[ ${model} == "bart" ]]; then
  init_ckpt=data/bart/bart.large/model.pt
elif [[ ${model} == "sealnq" ]]; then
  init_ckpt=data/SEAL-checkpoint+index.NQ/SEAL.NQ.pt
elif [[ ${model} == "sealkilt" ]]; then
  init_ckpt=data/SEAL-checkpoint+index.KILT/SEAL.KILT.pt
else
  exit
fi

data=$2  # This folder should contain the correct files if you have run scripts/training/preprocess_fairseq.sh before!
output=$3  # output directory

num_update=100000
save_every=10000
warmup=500
patience=10

fairseq-train \
  ${data}/bin \
  --finetune-from-model ${init_ckpt} \
  --save-dir ${output} \
  --arch bart_large \
  --task translation \
  --criterion label_smoothed_cross_entropy \
  --source-lang source --target-lang target \
  --truncate-source \
  --label-smoothing 0.1 \
  --max-tokens 4096 \
  --update-freq 1 \
  --max-update ${num_update} \
  --required-batch-size-multiple 1 \
  --save-interval-updates ${save_every} \
  --keep-interval-updates 3 \
  --dropout 0.1 \
  --attention-dropout 0.1 \
  --relu-dropout 0.0 \
  --weight-decay 0.01 \
  --optimizer adam \
  --adam-betas "(0.9, 0.999)" \
  --adam-eps 1e-08 \
  --clip-norm 0.1 \
  --lr-scheduler polynomial_decay \
  --lr 3e-05 \
  --total-num-update ${num_update} \
  --warmup-updates ${warmup} \
  --fp16 \
  --num-workers 10 \
  --no-epoch-checkpoints \
  --share-all-embeddings \
  --layernorm-embedding \
  --share-decoder-input-output-embed \
  --skip-invalid-size-inputs-valid-test \
  --log-format json \
  --log-interval 100 \
  --patience ${patience} \
  --find-unused-parameters
