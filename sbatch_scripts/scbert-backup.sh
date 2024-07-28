#!/bin/bash
#SBATCH -J scbert_job
#SBATCH -p gpu-a100-dev
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -o scbert_job.%j.out
#SBATCH -e scbert_job.%j.err
#SBATCH --mail-user=yichao1liu@gmail.com
#SBATCH --mail-type=begin
#SBATCH --mail-type=end



# 替换为您的conda安装路径
source /work/09735/yichao/ls6/miniconda/etc/profile.d/conda.sh
conda activate scbert

# 指定每个节点使用的GPU数量，假设为3
NPROC_PER_NODE=3

python -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --use_env \
    pretrain_modify.py --data_path "./data/panglao_human.h5ad"
