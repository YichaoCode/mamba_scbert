#!/bin/bash
#SBATCH --job-name=Local_PerformerTest    # 作业名称
#SBATCH --nodes=2                   # 请求的节点数
#SBATCH --ntasks-per-node=1         # 每个节点上的任务数
#SBATCH --output=PerformerTest_%j.out # 标准输出文件
#SBATCH --error=PerformerTest_%j.err  # 标准错误文件
#SBATCH -p gpu-a100-dev                 # 指定分区
#SBATCH -t 01:00:00                 # 时间限制
#SBATCH --mail-user=sharonhu54@gmail.com
#SBATCH --mail-type=begin               
#SBATCH --mail-type=end

# 初始化 Conda 环境
source /work/09735/yichao/ls6/miniconda/etc/profile.d/conda.sh

# 激活你的 Conda 环境
conda activate scbert

# 切换到特定的工作目录
cd /work/09735/yichao/ls6/jerry19970311/scBERT

# 获取主节点的地址
MASTER_ADDR=$(hostname)

# 执行PyTorch分布式训练命令
srun bash -c "torchrun --nproc_per_node=3 --nnodes=2 --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=29500 pretrain_modify.py --data_path='./data/panglao_human.h5ad'"
