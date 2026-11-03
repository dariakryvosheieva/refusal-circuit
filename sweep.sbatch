#!/bin/bash
#SBATCH --job-name=sweep_detector_layer
#SBATCH --partition=ou_bcs_normal
#SBATCH --account=mit_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
TAG="${TAG:-qwen1.5b_base}"
PROBE_STEPS="${PROBE_STEPS:-500}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p logs generations

export HF_HOME=/orcd/scratch/orcd/001/daria_k/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "model=$MODEL tag=$TAG gpus=${CUDA_VISIBLE_DEVICES:-none} node=$(hostname) extra=$EXTRA_ARGS"

eval /orcd/scratch/orcd/001/daria_k/.venv/bin/python sweep_detector_layer.py \
    --output sweep_detector_layer.pt
