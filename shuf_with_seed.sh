#!/usr/bin/env bash

file=$1

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

shuf --random-source=<(get_seeded_random 42) ${file}
