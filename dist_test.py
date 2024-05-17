import torch
import torch.distributed as dist
import socket

def main():
    print("Starting the main function.")
    backend = 'nccl'
    print(f"Initializing process group with backend '{backend}'.")

    # 初始化分布式环境
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_gpus = torch.cuda.device_count()

    # 为当前进程分配一个GPU
    device_id = rank % num_gpus
    torch.cuda.set_device(device_id)

    hostname = socket.gethostname()

    # 执行一个简单的Tensor运算
    tensor = torch.tensor([rank], device="cuda")
    result = tensor * 2  # 简单的计算，用于示例

    print(f"Rank {rank} on {hostname} using CUDA device {device_id} computed result: {result.item()}")

    # 在主节点上汇总和打印信息（可选）
    if rank == 0:
        print(f"Total number of processes: {world_size}")

if __name__ == "__main__":
    try:
        main()
        print("Script completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

