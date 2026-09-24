#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 0z5a
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

work=/home/gongji/0z5a-work/0z5a-ptx
cd "$work/src/cudnn-frontend-20f-localdeps"
export CUDA_VISIBLE_DEVICES=4
export LD_LIBRARY_PATH="$work/lib64:/home/gongji/0z5a/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

run_case() {
    local backend=$1 dtype=$2 dim=$3 sq=$4 skv=$5
    local tag="${dtype}-d${dim}-q${sq}-kv${skv}-${backend}"
    local log="$work/evidence620/bench-${tag}.log"
    "$work/bin/python" benchmark/attention_training/benchmark_single_sdpa.py \
        --sdpa_backend "$backend" --batch_size 1 --num_q_heads 8 --num_kv_heads 8 \
        --q_seqlen "$sq" --kv_seqlen "$skv" --head_dim "$dim" --data_type "$dtype" \
        --num_iterations 30 --num_warmup_iterations 8 --format_output --skip_ref --verbose \
        --case_tag "$tag" > "$log" 2>&1
    grep "^${tag}," "$log"
}

for backend in cudnn cudnn_oss; do
    run_case "$backend" float16 128 64 64
    run_case "$backend" float16 256 64 64
    run_case "$backend" bfloat16 128 128 256
    run_case "$backend" bfloat16 256 128 256
done
