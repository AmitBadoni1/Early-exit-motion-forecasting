#!/bin/bash
#SBATCH --job-name=simpl_train
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/storage/ice-shared/cs7643/group-a/simpl_train-%j.out

module load anaconda3

ulimit -SHn 51200
ulimit -s unlimited

cd "/storage/ice-shared/cs7643/group-a/Early-exits-in-motion-forecasting" || exit 1

PYTHON=/storage/ice-shared/cs7643/group-a/envs/simpl310/bin/python
export PYTHON

echo "==== ENV CHECK ===="
echo "Using PYTHON=$PYTHON"
$PYTHON -c "import sys; print(sys.executable)"
$PYTHON -c "import torch, numpy; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); print('NumPy:', numpy.__version__)"

echo "==== GPU INFO ===="
nvidia-smi || true

echo "==== START TRAINING ===="
bash scripts/simpl_av2_train_small.sh
