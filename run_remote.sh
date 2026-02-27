#!/bin/bash
#SBATCH --job-name=raytracing_hw01
#SBATCH --output=render_out.log
#SBATCH --error=render_err.log
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

#SBATCH --time=00:10:00
#SBATCH --mem=4G

PROJECT_ROOT="/home/zemanm40/ni-gpu/zemanm40"
HW_SCRIPT="$PROJECT_ROOT/hw01/main.py"

source "$PROJECT_ROOT/.venv/bin/activate"

# Set PYTHONPATH for'utils' folder/module
export PYTHONPATH="$PROJECT_ROOT"

"$PROJECT_ROOT"/.venv/bin/python "$HW_SCRIPT"
