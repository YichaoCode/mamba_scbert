#!/bin/bash

#SBATCH --job-name=scbert_dist # 作业名称
#SBATCH --nodes=2 # 请求的节点数
#SBATCH --output=sbatch_result/scbert_dist_%j_%x_%N_%P.out # 标准输出文件
#SBATCH --error=sbatch_result/scbert_dist_%j_%x_%N_%P.err # 标准错误文件
#SBATCH -p gpu-a100-dev # 指定分区
#SBATCH -t 00:30:00 # 时间限制
#SBATCH --mail-user=yichao1liu@gmail.com
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

# 初始化 Conda 环境
source /work/09735/yichao/ls6/miniconda/etc/profile.d/conda.sh

# 激活你的 Conda 环境
conda activate scbert

# 切换到特定的工作目录
cd /work/09735/yichao/ls6/dev/scBERT

# 获取主节点的地址
MASTER_ADDR=$(hostname)

# 执行PyTorch分布式训练命令
srun bash -c "torchrun --nproc_per_node=3 --nnodes=2 --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=29500 pretrain_modify.py --data_path='./data/panglao_human.h5ad'"