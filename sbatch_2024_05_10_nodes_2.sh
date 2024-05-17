#!/bin/bash
#SBATCH --job-name=scbert_dist
#SBATCH --nodes=2
#SBATCH -p gpu-a100-dev
#SBATCH --ntasks-per-node=3 # 可删
#SBATCH -t 00:10:00 
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
cmd="torchrun \
--nproc_per_node=3 \
--nnodes=2 \
--node_rank=\$SLURM_NODEID \
--master_addr=$MASTER_ADDR \
--master_port=29500 \
pretrain_2024-05-09_structured.py \
--data_path='./data/panglao_human.h5ad'"

srun bash -c "$cmd"