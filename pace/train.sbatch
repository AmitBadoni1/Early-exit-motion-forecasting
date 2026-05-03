#!/bin/bash

#SBATCH --job-name=simpl_train

#SBATCH --partition=ice-gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8

#SBATCH --mem=48G

#SBATCH --time=16:00:00

#SBATCH --output=/storage/ice-shared/cs7643/group-a/simpl_train_%j.out



module load anaconda3

source $(conda info --base)/etc/profile.d/conda.sh

conda activate /storage/ice-shared/cs7643/group-a/envs/simpl310



ulimit -SHn 51200

ulimit -s unlimited



cd "/storage/ice-shared/cs7643/group-a/Early-exits-in-motion-forecasting"



echo "==== ENV CHECK ===="

python -c "import torch, numpy; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); print('NumPy:', numpy.__version__)"



echo "==== GPU INFO ===="

nvidia-smi



echo "==== START TRAINING ===="

bash scripts/simpl_av2_train.sh
