#!/bin/bash
#SBATCH --job-name=raytracing
#SBATCH --output=render_out.log
#SBATCH --error=render_err.log
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

#SBATCH --time=00:10:00
#SBATCH --mem=32G

PROJECT_ROOT="/home/zemanm40/ni-gpu/zemanm40"
HW_SCRIPT="$PROJECT_ROOT/src/main.py"

source "$PROJECT_ROOT/.venv/bin/activate"

# Set PYTHONPATH for'utils' folder/module
export PYTHONPATH="$PROJECT_ROOT"

"$PROJECT_ROOT"/.venv/bin/python -OO "$HW_SCRIPT"
