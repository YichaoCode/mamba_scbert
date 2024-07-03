#!/bin/bash
#SBATCH --job-name=MambaTest_a100_15_3_24    # 作业名称
#SBATCH --nodes=15                   # 请求的节点数
#SBATCH --ntasks-per-node=1         # 每个节点上的任务数
#SBATCH --output=MambaTest_%j.out # 标准输出文件
#SBATCH --error=MambaTest_%j.err  # 标准错误文件
#SBATCH -p gpu-a100                 # 指定分区
#SBATCH -t 24:00:00                 # 时间限制
#SBATCH --mail-user=sharonhu54@gmail.com
#SBATCH --mail-type=begin               
#SBATCH --mail-type=end

# 初始化 Conda 环境
source /work/09735/yichao/ls6/miniconda/etc/profile.d/conda.sh

# 激活你的 Conda 环境
conda activate scbert6

# 切换到特定的工作目录
cd /work/09735/yichao/ls6/jerry19970311/scBERT

# 获取主节点的地址
MASTER_ADDR=$(hostname)

# 执行PyTorch分布式训练命令
# torchrun --nproc_per_node=1 --nnodes=1 --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=29500 pretrain_modify_mamba.py --batch_size 3 --data_path='./data/panglao_human.h5ad' 2>&1
# srun bash -c "torchrun --nproc_per_node=3 --nnodes=6 --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=29500 pretrain_modify_mamba.py --batch_size 3 --data_path='./data/panglao_human.h5ad' 2>&1"
srun bash -c "torchrun --nproc_per_node=3 --nnodes=15 --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=29500 pretrain_modify_mamba_continue.py --ckpt_source './ckpts/panglao_pretrain_15.pth' --ckpt_dir './ckpts2/' --batch_size 15 --data_path='./data/panglao_human.h5ad' 2>&1"
