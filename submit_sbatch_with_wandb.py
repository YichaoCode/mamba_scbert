import os
import subprocess
import logging


# 切换到指定的工作目录
os.chdir("/work/09735/yichao/ls6/dev/scBERT")
print("Current working directory:", os.getcwd())



# 配置日志记录
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===== 参数预设组合 =====
# 组合 1：单节点，开发分区，快速测试
# params = {
#     "job_name": "scbert_test",
#     "nodes": 1,
#     "partition": "gpu-a100-dev",
#     "time_limit": "00:30:00",
#     "mail_user": "yichao1liu@gmail.com",
#     "mail_type": "begin,fail,end",
#     "output_log": "./slurmlogs/%x_%j_nodes_%N.out",
#     "error_log": "./slurmlogs/%x_%j_nodes_%N.err",
#     "data_path": "./data/panglao_human.h5ad",
#     "model_type": "performer",
#     "nnodes": 1,
# }

# 组合 2：多节点，标准分区，常规训练
# params = {
#     "job_name": "scbert_train",
#     "nodes": 4,
#     "partition": "gpu-a100",
#     "time_limit": "08:00:00",
#     "mail_user": "yichao1liu@gmail.com",
#     "mail_type": "begin,fail,end",
#     "output_log": "./slurmlogs/%x_%j_nodes_%N.out",
#     "error_log": "./slurmlogs/%x_%j_nodes_%N.err",
#     "data_path": "./data/panglao_human.h5ad",
#     "model_type": "mamba",
#     "nnodes": 4,
# }

# 组合 3：高性能需求，15节点
# params = {
#     "job_name": "scbert_high_perf",
#     "nodes": 15,
#     "partition": "gpu-a100",
#     "time_limit": "24:00:00",
#     "mail_user": "yichao1liu@gmail.com",
#     "mail_type": "begin,fail,end",
#     "output_log": "./slurmlogs/%x_%j_nodes_%N.out",
#     "error_log": "./slurmlogs/%x_%j_nodes_%N.err",
#     "data_path": "./data/panglao_human.h5ad",
#     "model_type": "performer",
#     "nnodes": 15,
# }

# ===== 当前使用的参数组合 =====
# 复制您希望使用的组合并取消注释
params = {
    "job_name": "scbert_default",
    "nodes": 2,
    "partition": "gpu-a100-dev",
    "time_limit": "02:00:00",
    "mail_user": "yichao1liu@gmail.com",
    "mail_type": "begin,fail,end",
    "output_log": "./slurmlogs/%x_%j_nodes_%N.out",
    "error_log": "./slurmlogs/%x_%j_nodes_%N.err",
    "data_path": "./data/panglao_human.h5ad",
    "model_type": "mamba",
    "nnodes": 2,
}

def generate_slurm_script(params):
    """读取 SLURM 模板并根据参数生成 SLURM 脚本"""
    try:
        with open("sbatch_template.sh", "r") as f:
            slurm_script = f.read()
            for key, value in params.items():
                slurm_script = slurm_script.replace(f"{{{{{key}}}}}", str(value))
        
        generated_script_path = f"generated_{params['job_name']}.sh"
        with open(generated_script_path, "w") as f:
            f.write(slurm_script)
        
        logger.info("Generated SLURM script at %s", generated_script_path)
        return generated_script_path
    except Exception as e:
        logger.error("Error generating SLURM script: %s", e)
        raise

def submit_job(script_path):
    """使用 sbatch 提交 SLURM 作业"""
    try:
        logger.info("Submitting job using script: %s", script_path)
        result = subprocess.run(["sbatch", script_path], check=True, capture_output=True, text=True)
        logger.info("Job submitted successfully: %s", result.stdout.strip())
    except subprocess.CalledProcessError as e:
        logger.error("Failed to submit job: %s", e.stderr)
        raise

def main():
    # 使用当前的参数组合
    script_path = generate_slurm_script(params)
    submit_job(script_path)

if __name__ == "__main__":
    main()
