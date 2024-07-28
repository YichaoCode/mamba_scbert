#!/bin/bash
#SBATCH -J scbert_job
#SBATCH -p gpu-a100-dev       # 指定分区
#SBATCH -t 00:10:00           # 运行时间
#SBATCH -N 2                  # 请求2个节点
#SBATCH -n 256                  # 总共请求6个任务（因为 2个节点 x 每个节点3个任务）
#SBATCH --ntasks-per-node=128   # 每个节点3个任务
#SBATCH -o scbert_job.%j.out  # 标准输出文件
#SBATCH -e scbert_job.%j.err  # 标准错误文件
#SBATCH --mail-user=yichao1liu@gmail.com
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

# 加载模块
source /work/09735/yichao/ls6/miniconda/etc/profile.d/conda.sh
conda activate scbert

# 检查端口是否可用的函数
check_port() {
    local port=$1
    nc -z localhost $port &> /dev/null
    if [ $? -eq 0 ]; then
        return 1 # 端口已被占用
    else
        return 0 # 端口可用
    fi
}

# 主执行函数
main() {
    echo "当前工作目录: $(pwd)"
    # nvidia-smi


    # 获取 SLURM 相关信息
    echo "SLURM_JOB_ID: $SLURM_JOB_ID"
    echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
    echo "SLURM_NODEID: $SLURM_NODEID"


    echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
    echo "SLURM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE"
    echo "SLURM_NODELIST: $SLURM_NODELIST"
    echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
    echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"
    echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
    echo "SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"

    # 设置参数
    local nproc_per_node=3
    local nnodes=$SLURM_JOB_NUM_NODES
    local node_rank=$SLURM_NODEID

    # 执行分布式训练前的额外检查
    # echo "检查 SLURM 分配的 GPU:"
    # srun --exclusive --ntasks=1 nvidia-smi
    # srun --exclusive --nodes=$SLURM_JOB_NUM_NODES nvidia-smi


    # 获取主节点的主机名
    local master_node=$(srun --nodes=1 --ntasks=1 hostname)
    echo "主节点主机名: $master_node"


    # 获取一个随机端口号
    # local master_port=$(( RANDOM % 10000 + 20000 ))
    # local master_port=34568
    # echo "主节点端口号: $master_port"

    # 端口选择
    local base_port=34568
    local port_offset=0
    local selected_port=0
    while [ $port_offset -lt 10 ]; do
        selected_port=$((base_port + port_offset))
        check_port $selected_port
        if [ $? -eq 0 ]; then
            echo "选择端口: $selected_port"
            break
        fi
        port_offset=$((port_offset + 1))
    done
    if [ $port_offset -eq 10 ]; then
        echo "错误: 无法找到可用端口。"
        exit 1
    fi
    
    # 执行分布式训练
    echo "开始执行分布式训练.."
    srun torchrun \
        --nproc_per_node=$SLURM_NTASKS_PER_NODE \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --node_rank=$SLURM_NODEID \
        --master_addr=$master_node \
        --master_port=$selected_port \
	pretrain_modify.py --data_path "./data/panglao_human.h5ad"
}

# 调用主执行函数
main

