#!/bin/bash

#SBATCH --job-name=av2_preproc

#SBATCH --cpus-per-task=12

#SBATCH --mem=64G

#SBATCH --time=04:00:00

#SBATCH --output=/storage/ice-shared/cs7643/group-a/av2_preproc_%j.out


module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /storage/ice-shared/cs7643/group-a/envs/simpl310



ulimit -SHn 51200

ulimit -s unlimited


cd "/storage/ice-shared/cs7643/group-a/Early-exits-in-motion-forecasting/scripts"



bash av2_preproc_all.sh

