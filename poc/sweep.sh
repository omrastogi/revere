#!/bin/bash
# POC Slurm chain.  bash sweep.sh emb           -> embedding arrays (V100-SXM2 / T4 / H200) + merge
#                   bash sweep.sh heads [DEP]   -> sweepA -> finalA -> sweepB -> finalB -> report
set -e
cd /projects/adsum-mllm-om/ucf_anomaly/poc && mkdir -p logs results ckpt emb
PY="/projects/adsum-mllm-om/miniconda3/envs/mllm3/bin/python -I"
H5=/projects/adsum-mllm-om/data/ucf_crime_h5/ucf_crime_224_s4.h5
SHARDS=32 NT=3  # GPU partition allows 8 queued+running jobs per user: 3+1+3+1

if [ "$1" = emb ]; then
  ids=""
  for g in v100-sxm2:8 t4:4 h200:4; do  # one array per GPU type; every task claims free shards via lock files
    id=$(sbatch --parsable -p gpu -t 3:00:00 -c 6 --mem=48G --gres=gpu:${g%%:*}:1 --array=0-$((${g##*:} - 1)) \
         -J poc_emb -o logs/emb_%A_%a.log --wrap "$PY extract_emb.py run $H5 emb $SHARDS 5") || continue
    ids="$ids:$id"; echo "emb ${g%%:*}: $id"
  done
  echo "merge: $(sbatch --parsable --dependency=afterany$ids -p short -t 0:30:00 -c 2 --mem=32G -J poc_merge \
       -o logs/merge_%j.log --wrap "$PY extract_emb.py merge $H5 emb $SHARDS")"
fi

if [ "$1" = heads ]; then
  GPU="-p gpu -t 2:00:00 -c 4 --mem=32G --gres=gpu:v100-sxm2:1"
  a=$(sbatch --parsable ${2:+--dependency=afterok:$2} $GPU --array=0-$((NT - 1)) -J poc_sweepA -o logs/sweepA_%A_%a.log \
      --wrap "$PY train_heads.py sweepA \$SLURM_ARRAY_TASK_ID $NT")
  fa=$(sbatch --parsable --dependency=afterok:$a $GPU -J poc_finalA -o logs/finalA_%j.log --wrap "$PY train_heads.py finalA")
  b=$(sbatch --parsable --dependency=afterok:$fa $GPU --array=0-$((NT - 1)) -J poc_sweepB -o logs/sweepB_%A_%a.log \
      --wrap "$PY train_heads.py sweepB \$SLURM_ARRAY_TASK_ID $NT")
  fb=$(sbatch --parsable --dependency=afterok:$b $GPU -J poc_finalB -o logs/finalB_%j.log --wrap "$PY train_heads.py finalB")
  r=$(sbatch --parsable --dependency=afterok:$fb -p short -t 0:10:00 -c 1 --mem=4G -J poc_report -o logs/report_%j.log \
      --wrap "$PY report.py")
  echo "sweepA=$a finalA=$fa sweepB=$b finalB=$fb report=$r"
fi
