#!/bin/bash
#SBATCH -J pretrain_mamba
#SBATCH -o pretrain_mamba.o%j
#SBATCH -e pretrain_mamba.e%j
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 01:00:00

pip install pandas
conda activate scbert
torchrun --nnodes=1 --nproc_per_node=3 pretrain_mamba_debug.py --data_path "./data/panglao_human.h5ad" 2>&1 | tee $WORK/jerry19970311/scBERT/errors/2024022401.log
