#!/bin/bash
#SBATCH --job-name=sc_finetune_prod
#SBATCH --nodes=4
#SBATCH -p gpu-a100 
#SBATCH -t 24:00:00 
#SBATCH --mail-user=daizhilian@hotmail.com
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
srun bash -c "torchrun --nproc_per_node=3 --nnodes=4 --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=29500 finetune_zl.py --model_path='./tx_pretrain.pth' --data_path='./data/Zheng68K.h5ad'"


