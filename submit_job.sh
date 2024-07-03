#!/bin/bash

# 默认参数值
N_LAYER=${1:-5}
N_DIM=${2:-4000}
NUM_NODES=${3:-8}
BATCH_SIZE=${4:-3}

# 构建作业名称
JOB_NAME="Mamba_L${N_LAYER}_D${N_DIM}_N${NUM_NODES}_B${BATCH_SIZE}"

# 提交作业
sbatch --job-name="$JOB_NAME" \
       --nodes=$NUM_NODES \
       --export=ALL,N_LAYER=$N_LAYER,N_DIM=$N_DIM,NUM_NODES=$NUM_NODES,BATCH_SIZE=$BATCH_SIZE \
       sbatch_yichao.sh

echo "Submitted job: $JOB_NAME"
