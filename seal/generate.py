# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List
import argparse
import random
import json
import os
import tqdm
from more_itertools import chunked
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from seal import fm_index_generate
from seal.retrieval import SEALSearcher

def format_example_for_generation(example: Dict, context_type: str = "positive_ctxs", num_contexts: int = 3) -> str:
    input_to_model = []
    for i in range(min(len(example[context_type]), num_contexts)):
        title, text = example[context_type][i]["title"], example[context_type][i]["text"]
        input_to_model.append(f"{title}: {text} | ")
    input_to_model.append(example["question"])
    input_to_model = " ".join(input_to_model)
    return input_to_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, metavar="path", help="Path to input file.")
    parser.add_argument("--output", type=str, metavar="path", help="Path to output file.")
    parser.add_argument("--context_type", type=str, choices=["positive_ctxs", "ctxs"], default="positive_ctxs", help="Context to use.")
    parser.add_argument("--model_class", type=str, choices=["seal", "bart"], default="seal", help="Model class.")
    parser.add_argument("--max_length", type=int, default=128, help="Max length of the generation.")
    parser.add_argument("--debug", action="store_true", help="Debug using a small number of inputs.")
    SEALSearcher.add_args(parser)
    args = parser.parse_args()
    print(args)

    random.seed(2022)
    
    # multi-job setup
    is_slurm = os.getenv("SLURM_JOB_ID") is not None
    if is_slurm:
        world_size = int(os.getenv("SLURM_NTASKS"))
        local_rank = int(os.getenv("SLURM_LOCALID"))
        global_rank = int(os.getenv("SLURM_PROCID"))
        args.device = f"cuda:{local_rank}"
        print(f"SLURM job rank {global_rank} with GPU device {args.device}")
    else:
        world_size = 1
        local_rank = global_rank = 0
    is_multi_jobs = world_size > 1
    
    # load all data
    with open(args.input, "r") as fin:
        data: List[Dict] = json.load(fin)
        
    # slice data for current job
    if is_multi_jobs:
        size_per_job = int(np.ceil(len(data) / world_size))
        start, end = global_rank * size_per_job, min((global_rank + 1) * size_per_job, len(data))
        data = data[start:end]
        print(f"rank {global_rank}: #examples {len(data)} from {start} to {end}")
    else:
        print(f"rank {global_rank}: #examples {len(data)} from {0} to {len(data)}")
    
    # format data
    inputs: List[str] = [format_example_for_generation(example, context_type=args.context_type) for example in data]
    
    if args.debug:
        data = data[:5]
        inputs = inputs[:5]
    
    # load model
    index = None
    if args.model_class == "seal":
        searcher = SEALSearcher.from_args(args)
        tokenizer = searcher.bart_tokenizer
        model = searcher.bart_model
    elif args.model_class == "bart":
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")
        model.to(args.device)
    else:
        raise NotImplementedError
    
    # generate and save outputs
    args.output = f"{args.output}.{global_rank}" if is_multi_jobs else args.output
    with tqdm.tqdm(total=len(inputs), desc="Generating", disable=False) as bar, open(args.output, "w") as fout:
        batches = chunked(inputs, args.batch_size)
        for batch in batches:        
            batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
            batch = {k: v.to(args.device) for k, v in batch.items()}
            
            outputs = fm_index_generate(
                model,
                index,
                **batch,
                keep_history=False,
                transformers_output=True,
                always_allow_eos=True,
                max_length=args.max_length,
                num_beams=args.beam,
                disable_fm_index=True)
            outputs = outputs[::args.beam]
            for out in outputs:
                fout.write(tokenizer.decode(out, skip_special_tokens=True).strip() + "\n")
            bar.update(len(outputs))
