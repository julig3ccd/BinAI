#!/bin/bash

#SBATCH -J binai-train
#SBATCH -p NvidiaAll
#SBATCH -t 10:00:00

mkdir -p "$HOME/slurm-logs" "$HOME/results"
LOG="${HOME}/slurm-logs/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-noid}.log"
exec > >(tee -a "$LOG") 2>&1

set -Eeuo pipefail
trap 'st=$?; echo "ERR on line $LINENO: $BASH_COMMAND (exit $st)"; exit $st' ERR

echo "==== START $(date) node=$(hostname) job=$SLURM_JOB_ID ===="
echo "HOME=$HOME  PWD=$(pwd)  TMPDIR=${TMPDIR-<unset>}"

# Activate virtual environment
set +u
source "$PROJECT_DIR/binai_venv/bin/activate"
set -u

python -V || true

# Set up paths
PROJECT_DIR="$HOME/BinAI"
SRC_DATA="$HOME/data/final_preprocessed"
[ -d "$SRC_DATA" ] || { echo "ERROR: SRC_DATA not found: $SRC_DATA"; exit 2; }

# Copy dataset to local disk of computation node for faster access
JOB_LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
JOB_TMP="$JOB_LOCAL_BASE/binai_${SLURM_JOB_ID}"
mkdir -p "$JOB_TMP"; chmod 700 "$JOB_TMP"; export TMPDIR="$JOB_TMP"
echo "JOB_TMP=$JOB_TMP"
df -h "$JOB_LOCAL_BASE" | sed -n '1,2p'

DST_DATA="$JOB_TMP/final_preprocessed"
cp -r "$SRC_DATA" "$DST_DATA"
echo "DATA_DST=$(readlink -f "$DST_DATA")"
ls -la "$DST_DATA"

# Change to project directory
cd "$PROJECT_DIR"

# Run training script with final_preprocessed data
python train_base.py \
    --train_data "$DST_DATA/train" \
    --val_data "$DST_DATA/val" \
    --test_data "$DST_DATA/test" \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.0001 \
    --num_workers 4 \
    --num_hidden_layers 12 \
    --num_heads 12 \
    --out_dir "$PROJECT_DIR/output/slurm_run_${SLURM_JOB_ID}" \
    --create_opcode_ids

echo "==== END $(date) ===="
