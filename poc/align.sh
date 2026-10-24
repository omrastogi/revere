#!/bin/bash
# Phase-2 alignment (align.py), run from ucf_anomaly/ after `python poc/align.py text`.
#   bash poc/align.sh clips        -> clip-embedding jobs on every GPU type at once; they share vjepa_emb/part_XX via locks
#   bash poc/align.sh train DEPS   -> train + eval after the clip jobs (DEPS = colon-separated job ids)
set -e
U=/projects/adsum-mllm-om/ucf_anomaly
PY="/projects/adsum-mllm-om/miniconda3/envs/mllm3/bin/python -I"
A=$U/poc/align.py
cd $U && mkdir -p logs vjepa_emb
if [ "$1" = clips ]; then
  ids=""
  for g in h200:2 h100:1 a100:2 l40s:2; do  # sharing partition: 1 h limit, starts fast
    id=$(sbatch --parsable -p sharing --gres=gpu:${g%%:*}:1 --array=0-$((${g##*:} - 1)) -t 1:00:00 -c 12 --mem=48G \
         -J align_clips -o logs/align_clips_%A_%a.log --wrap "$PY $A clips") && ids="$ids:$id"
  done
  id=$(sbatch --parsable -p gpu --gres=gpu:v100-sxm2:1 --array=0-3 -t 2:00:00 -c 12 --mem=48G --export=ALL,CLAIM_MIN=100 \
       -J align_clips -o logs/align_clips_%A_%a.log --wrap "$PY $A clips") && ids="$ids:$id"
  echo "clip jobs: ${ids#:}"
fi
if [ "$1" = train ]; then
  sbatch --parsable ${2:+--dependency=afterany:$2} -p sharing --gres=gpu:h200:1 -t 1:00:00 -c 8 --mem=64G \
    -J align_train -o logs/align_train_%j.log --wrap "set -e; $PY $A train; $PY $A eval"
fi
