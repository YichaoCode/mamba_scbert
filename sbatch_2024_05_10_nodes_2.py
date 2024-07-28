import os
import subprocess
from clearml import Task

# 初始化 ClearML 任务
task = Task.init(project_name='scBERT_mamba_plus', task_name='scbert_dist')

# 设置SLURM环境变量
os.environ['SBATCH_JOB_NAME'] = 'scbert_dist'
os.environ['SBATCH_NODES'] = '2'
os.environ['SBATCH_PARTITION'] = 'gpu-a100-dev'
os.environ['SBATCH_TIME'] = '00:30:00'
os.environ['SBATCH_MAIL_USER'] = 'yichao1liu@gmail.com'
os.environ['SBATCH_MAIL_TYPE'] = 'begin,fail,end'
os.environ['SBATCH_OUTPUT'] = './slurmlogs/%x_%j_nodes_%N.out'
os.environ['SBATCH_ERROR'] = './slurmlogs/%x_%j_nodes_%N.err'

# 创建sbatch命令
sbatch_cmd = [
    'sbatch',
    '--job-name', os.environ['SBATCH_JOB_NAME'],
    '--nodes', os.environ['SBATCH_NODES'],
    '--partition', os.environ['SBATCH_PARTITION'],
    '--time', os.environ['SBATCH_TIME'],
    '--mail-user', os.environ['SBATCH_MAIL_USER'],
    '--mail-type', os.environ['SBATCH_MAIL_TYPE'],
    '--output', os.environ['SBATCH_OUTPUT'],
    '--error', os.environ['SBATCH_ERROR']
]

# 执行sbatch命令
subprocess.run(sbatch_cmd)

# 初始化Conda环境
conda_sh = '/work/09735/yichao/ls6/miniconda/etc/profile.d/conda.sh'
subprocess.run(['source', conda_sh], shell=True, executable='/bin/bash')

# 激活你的Conda环境
subprocess.run(['conda', 'activate', 'scbert'], shell=True, executable='/bin/bash')

# 切换到特定的工作目录
os.chdir('/work/09735/yichao/ls6/dev/scBERT')

# 获取主节点的地址
master_addr = subprocess.check_output(['hostname']).strip().decode()

# 创建PyTorch分布式训练命令
cmd = f"""
torchrun \\
--nproc_per_node=3 \\
--nnodes=2 \\
--node_rank=$SLURM_NODEID \\
--master_addr={master_addr} \\
--master_port=29500 \\
pretrain_2024-05-13_distributed.py \\
--data_path='./data/panglao_human.h5ad'
"""

# 执行srun命令
srun_cmd = ['srun', 'bash', '-c', cmd]
subprocess.run(srun_cmd, shell=True, executable='/bin/bash')