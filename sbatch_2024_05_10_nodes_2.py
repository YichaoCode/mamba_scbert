import os
import subprocess
from clearml import Task

# 初始化 ClearML 任务并禁用 GPU 监控
task = Task.init(project_name='scBERT_mamba_plus', task_name='scbert_dist', auto_connect_arg_parser=False)
task.set_base_task(Task.get_task(project_name="scBERT_mamba_plus", task_name="scbert_dist", task_id="06a04d189b114cfe82356b557d0547e5"))

# 禁用 GPU 监控
task.set_system_tags(["disable_monitoring_gpu"])

# 设置 SLURM 环境变量
os.environ['SBATCH_JOB_NAME'] = 'scbert_dist'
os.environ['SBATCH_NODES'] = '2'
os.environ['SBATCH_PARTITION'] = 'gpu-a100-dev'
os.environ['SBATCH_TIME'] = '00:30:00'
os.environ['SBATCH_MAIL_USER'] = 'yichao1liu@gmail.com'
os.environ['SBATCH_MAIL_TYPE'] = 'begin,fail,end'
os.environ['SBATCH_OUTPUT'] = './slurmlogs/%x_%j_nodes_%N.out'
os.environ['SBATCH_ERROR'] = './slurmlogs/%x_%j_nodes_%N.err'

# 创建 sbatch 命令
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

# 执行 sbatch 命令并捕获作业 ID
result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
job_id = None
if result.returncode == 0:
    output = result.stdout
    for line in output.split('\n'):
        if "Submitted batch job" in line:
            job_id = line.split()[-1]
            print(f"Submitted batch job {job_id}")
else:
    print(f"Failed to submit job: {result.stderr}")
    exit(1)

# 等待作业完成
wait_cmd = ['squeue', '--job', job_id]
while True:
    result = subprocess.run(wait_cmd, capture_output=True, text=True)
    if job_id not in result.stdout:
        break

# 初始化 Conda 环境
conda_sh = '/work/09735/yichao/ls6/miniconda/etc/profile.d/conda.sh'
subprocess.run(['source', conda_sh], shell=True, executable='/bin/bash')

# 激活你的 Conda 环境
subprocess.run(['conda', 'activate', 'scbert'], shell=True, executable='/bin/bash')

# 切换到特定的工作目录
os.chdir('/work/09735/yichao/ls6/dev/scBERT')

# 获取主节点的地址
master_addr = subprocess.check_output(['hostname']).strip().decode()

# 创建 PyTorch 分布式训练命令
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

# 执行 srun 命令
srun_cmd = ['srun', 'bash', '-c', cmd]
subprocess.run(srun_cmd, shell=True, executable='/bin/bash')

# 读取并输出日志文件的内容
output_log = os.environ['SBATCH_OUTPUT'].replace('%x', os.environ['SBATCH_JOB_NAME']).replace('%j', job_id).replace('%N', 'master')
error_log = os.environ['SBATCH_ERROR'].replace('%x', os.environ['SBATCH_JOB_NAME']).replace('%j', job_id).replace('%N', 'master')

with open(output_log, 'r') as f:
    print(f"--- Standard Output ---\n{f.read()}")

with open(error_log, 'r') as f:
    print(f"--- Standard Error ---\n{f.read()}")