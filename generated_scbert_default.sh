#!/bin/bash
#SBATCH --job-name=scbert_default
#SBATCH --nodes=1
#SBATCH -p gpu-a100-dev
#SBATCH -t 02:00:00
#SBATCH --mail-user=yichao1liu@gmail.com
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --output=./slurmlogs/%x_%j_nodes_%N.out
#SBATCH --error=./slurmlogs/%x_%j_nodes_%N.err

# Initialize Conda environment
source /work/09735/yichao/ls6/miniconda/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate scbert

# Change to the specified working directory
cd /work/09735/yichao/ls6/dev/scBERT

# Get the master node's address
MASTER_ADDR=$(hostname)

# Execute the PyTorch distributed training command
srun bash -c "torchrun --nproc_per_node=3 --nnodes=1 --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=29500 \
pretrain_wandb_10_02.py --data_path='./data/panglao_human.h5ad' --model_type=performer"
