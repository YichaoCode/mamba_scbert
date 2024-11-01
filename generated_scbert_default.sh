#!/bin/bash
#SBATCH --job-name=scbert_default             # 作业名称
#SBATCH --nodes=1                   # 节点数
#SBATCH -p gpu-a100-dev                    # 分区
#SBATCH -t 02:00:00                   # 作业运行时间
#SBATCH --mail-user=yichao1liu@gmail.com           # 邮件通知用户
#SBATCH --mail-type=begin                   # 任务开始时发送邮件
#SBATCH --mail-type=fail                    # 任务失败时发送邮件
#SBATCH --mail-type=end                     # 任务结束时发送邮件
#SBATCH --output=./slurmlogs/%x_%j_nodes_%N.out # 标准输出文件
#SBATCH --error=./slurmlogs/%x_%j_nodes_%N.err  # 错误输出文件

# 初始化 Conda 环境
source /work/09735/yichao/ls6/miniconda/etc/profile.d/conda.sh

# 激活您的 Conda 环境
conda activate scbert

# 切换到指定的工作目录
cd /work/09735/yichao/ls6/dev/scBERT

# 获取主节点的地址
MASTER_ADDR=$(hostname)

# 执行 PyTorch 分布式训练命令
srun bash -c "torchrun --nproc_per_node=3 --nnodes=1 --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=29500 \
pretrain_wandb_10_02.py --data_path='./data/panglao_human.h5ad' --model_type=performer"

