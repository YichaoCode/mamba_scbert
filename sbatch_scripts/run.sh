#!/bin/bash

HOST=$1
NODES=$2
LOCAL_RANK=${PMI_RANK}

echo "HOST: $HOST"
echo "NODES: $NODES"
echo "LOCAL_RANK: $LOCAL_RANK"


# torchrun --nproc_per_node=3 --nnodes=2  --node_rank=0 --master_addr=c301-002 --master_port=29500 pretrain_modify.py --data_path "./data/panglao_human.h5ad"
# torchrun --nproc_per_node=3 --nnodes=2  --node_rank=1 --master_addr=c301-002 --master_port=29500 pretrain_modify.py --data_path "./data/panglao_human.h5ad"


torchrun --nproc_per_node=3 --nnodes=2  --node_rank=${LOCAL_RANK} --master_addr=c301-003 --master_port=29500 pretrain_modify.py --data_path "./data/panglao_human.h5ad"



# ibrun -np 6 ./run.sh c301-003 2
