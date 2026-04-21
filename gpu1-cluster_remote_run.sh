#!/bin/bash
#SBATCH --job-name=raytracing
#SBATCH --output=render_out.log
#SBATCH --error=render_err.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100_40:1  # requesting an A100 (40GB)

#SBATCH --time=00:30:00
#SBATCH --mem=32G

PROJECT_ROOT="/home/zemanm40/ni-gpu/zemanm40"
HW_SCRIPT="$PROJECT_ROOT/src/main.py"
source "$PROJECT_ROOT/.venv/bin/activate"

# Export project root so 'utils' can be found
export PYTHONPATH="$PROJECT_ROOT"

# Run the script
python "$HW_SCRIPT"
