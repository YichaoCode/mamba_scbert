#!/bin/bash
#SBATCH --job-name=scbert_dist
#SBATCH --nodes=16
#SBATCH -p gpu-a100 
#SBATCH -t 24:00:00 
#SBATCH --mail-user=yichao1liu@gmail.com
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --output=./slurmlogs/%x_%j_nodes_%N.out 
#SBATCH --error=./slurmlogs/%x_%j_nodes_%N.err 


# 初始化 Conda 环境
source /work/09735/yichao/ls6/miniconda/etc/profile.d/conda.sh

# 激活你的 Conda 环境
conda activate scbert

# 切换到特定的工作目录
cd /work/09735/yichao/ls6/dev/scBERT

# 获取主节点的地址
MASTER_ADDR=$(hostname)

# 执行PyTorch分布式训练命令
srun bash -c "torchrun --nproc_per_node=3 --nnodes=16 --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=29500 pretrain_2024-05-13_distributed.py --data_path='./data/panglao_human.h5ad'"
# srun bash -c "torchrun --nproc_per_node=3 --nnodes=15 --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=29500 pretrain_modify.py --data_path='./data/panglao_human.h5ad'"