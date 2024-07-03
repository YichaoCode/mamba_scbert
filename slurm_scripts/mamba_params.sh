#!/bin/bash
#SBATCH --job-name=Mamba_pretrain    # 作业名称
#SBATCH --nodes=8                   # 请求的节点数
#SBATCH --ntasks-per-node=1         # 每个节点上的任务数
#SBATCH --output=logs/Mamba_pretrain_%j.out # 标准输出文件
#SBATCH --error=logs/Mamba_pretrain_%j.err  # 标准错误文件
#SBATCH -p gpu-a100                 # 指定分区
#SBATCH -t 48:00:00                 # 时间限制
#SBATCH --mail-user=daizhilian@hotmail.com
#SBATCH --mail-type=begin               
#SBATCH --mail-type=end

# 初始化 Conda 环境
source /work/09735/yichao/ls6/miniconda/etc/profile.d/conda.sh

# 激活你的 Conda 环境
conda activate scbert6

# 切换到特定的工作目录
cd /work/09735/yichao/ls6/zhilian/jerry_mamba

# 获取主节点的地址
MASTER_ADDR=$(hostname)

n_layer=5
n_dim=800
# 执行PyTorch分布式训练命令
# torchrun --nproc_per_node=1 --nnodes=1 --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=29500 pretrain_modify_mamba.py --batch_size 3 --data_path='./data/panglao_human.h5ad' 2>&1
srun bash -c "torchrun --nproc_per_node=3 --nnodes=8 --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=29500 pretrain_modify_mamba_params.py --batch_size 3 --n_layer=$n_layer --n_dim=$n_dim --data_path='./data/panglao_human.h5ad' --ckpt_dir='./ckpts/mamba_${n_layer}_${n_dim}' 2>&1"
# srun bash -c "torchrun --nproc_per_node=3 --nnodes=15 --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=29500 pretrain_modify_mamba_continue.py --ckpt_source './ckpts/panglao_pretrain_15.pth' --ckpt_dir './ckpts2/' --batch_size 3 --data_path='./data/panglao_human.h5ad' 2>&1"
