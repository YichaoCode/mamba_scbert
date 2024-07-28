#!/bin/bash
#SBATCH -J scbert_job
#SBATCH -p gpu-a100-dev
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -o scbert_job.%j.out
#SBATCH -e scbert_job.%j.err
#SBATCH --mail-user=yichao1liu@gmail.com
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

# 加载模块
source /work/09735/yichao/ls6/miniconda/etc/profile.d/conda.sh
conda activate scbert

# 打印环境信息函数
print_environment_info() {
    # echo "环境变量:"
    # env
    echo "当前工作目录: $(pwd)"
    # echo "工作目录内容:"
    # ls -l
    nvidia-smi
}

# 检查文件存在性函数
check_file_existence() {
    local file_path=$1
    if [ -f "$file_path" ]; then
        echo "文件存在: $file_path"
    else
        echo "文件不存在: $file_path"
        exit 1
    fi
}

# 主执行函数
main() {
    print_environment_info
    check_file_existence "./data/gene2vec_16906.npy"
    
    # 获取 SLURM 相关信息
    echo "SLURM_JOB_ID: $SLURM_JOB_ID"
    echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
    echo "SLURM_NODEID: $SLURM_NODEID"

    # 设置参数
    local nproc_per_node=3
    local nnodes=$SLURM_JOB_NUM_NODES
    local node_rank=$SLURM_NODEID


    # 执行分布式训练前的额外检查
    echo "检查 SLURM 分配的 GPU:"
    srun --exclusive --ntasks=1 nvidia-smi


    # 执行分布式训练
    echo "开始执行分布式训练..."
    python -m torch.distributed.launch \
        --nproc_per_node=$nproc_per_node \
        --nnodes=$nnodes \
        --node_rank=$node_rank \
        --use_env \
        pretrain_modify.py --data_path "./data/panglao_human.h5ad"
}

# 调用主执行函数
main
