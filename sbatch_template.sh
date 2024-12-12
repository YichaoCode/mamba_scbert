#!/bin/bash
#SBATCH --job-name={{job_name}}
#SBATCH --nodes={{nodes}}
#SBATCH -p {{partition}}
#SBATCH -t {{time_limit}}
#SBATCH --mail-user={{mail_user}}
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --output=./slurmlogs/%x_%j_nodes_%N.out
#SBATCH --error=./slurmlogs/%x_%j_nodes_%N.err

# Initialize Conda environment
source /work/09735/yichao/ls6/miniconda/etc/profile.d/conda.sh

# Activate your Conda environment
# conda activate scbert
conda activate scbert6


# Change to the specified working directory
cd /work/09735/yichao/ls6/dev/scBERT

# Get the master node's address
MASTER_ADDR=$(hostname)

# Execute the PyTorch distributed training command
srun bash -c "torchrun --nproc_per_node=3 --nnodes={{nnodes}} --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=29500 \
pretrain_wandb_10_02.py --data_path='{{data_path}}' --model_type={{model_type}}"
