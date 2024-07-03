#!/bin/bash

#SBATCH --job-name=Mamba_pretrain
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -p gpu-a100
#SBATCH --time=48:00:00
#SBATCH --mail-user=yichao1liu@gmail.com
#SBATCH --mail-type=BEGIN,END

# 参数值（可以通过环境变量覆盖）
N_LAYER=${N_LAYER:-5}
N_DIM=${N_DIM:-4000}
NUM_NODES=${SLURM_JOB_NUM_NODES:-8}
BATCH_SIZE=${BATCH_SIZE:-3}

# 配置参数
CONDA_ENV="scbert6"
WORK_DIR="/work/09735/yichao/ls6/dev/20240630-dev-mamba_scbert"
DATA_PATH="./data/panglao_human.h5ad"

# 定义日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 初始化环境
init_env() {
    log "Initializing environment"
    source /work/09735/yichao/ls6/miniconda/etc/profile.d/conda.sh
    conda activate $CONDA_ENV
    cd $WORK_DIR
}

# 获取主节点地址
get_master_addr() {
    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    log "Master node address: $MASTER_ADDR"
}

# 执行训练
run_training() {
    log "Starting training with n_layer=$N_LAYER, n_dim=$N_DIM, nodes=$NUM_NODES, batch_size=$BATCH_SIZE"
    srun bash -c "torchrun \
        --nproc_per_node=3 \
        --nnodes=$NUM_NODES \
        --node_rank=\$SLURM_PROCID \
        --master_addr=$MASTER_ADDR \
        --master_port=29500 \
        pretrain_modify_mamba_params.py \
        --batch_size $BATCH_SIZE \
        --n_layer $N_LAYER \
        --n_dim $N_DIM \
        --data_path '$DATA_PATH' \
        --ckpt_dir './ckpts/mamba_${N_LAYER}_${N_DIM}' \
        $RESUME_FLAG \
        2>&1"
}

# 主函数
main() {
    log "Job started"
    log "Conda environment: $CONDA_ENV"
    log "Working directory: $WORK_DIR"
    log "Data path: $DATA_PATH"
    log "Number of nodes: $NUM_NODES"
    log "Number of layers: $N_LAYER"
    log "Dimension: $N_DIM"
    log "Batch size: $BATCH_SIZE"
    
    init_env
    get_master_addr
    run_training
    
    log "Job completed"
}

# 执行主函数
main
