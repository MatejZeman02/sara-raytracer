#!/bin/bash
#SBATCH --job-name=rt-gpu-analysis
#SBATCH --output=benchmark_logs/analysis_%j.out
#SBATCH --error=benchmark_logs/analysis_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100_40:1
#SBATCH --time=00:45:00
#SBATCH --mem=32G

# setup environment
project_root="/home/zemanm40/ni-gpu/zemanm40"
source "$project_root/.venv/bin/activate"
export PYTHONPATH="$project_root"

python "$project_root/benchmarks.py"
