import wandb
import logging
import time
import socket
import ssl
import requests

import os
os.environ['WANDB_DISABLE_GIT'] = 'true'

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def log_with_timestamp(message):
    """生成带时间戳的日志"""
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logger.info(f"{current_time} - {message}")

def measure_network_latency():
    """测量 DNS 解析、TCP 连接、SSL 握手的时间"""
    log_with_timestamp("🌐 测试与 W&B 的网络连通性...")
    url = "https://api.wandb.ai"
    
    # DNS 解析和连接时间
    start_time = time.time()
    try:
        response = requests.get(url, timeout=10)
        dns_and_connect_time = time.time() - start_time
        logger.info(f"✅ 网络连接正常，耗时：{dns_and_connect_time:.2f} 秒")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ 无法连接到 W&B API：{e}")
        return

def test_wandb_connectivity():
    """测试 W&B Python SDK 初始化和连接"""
    log_with_timestamp("🛠️ 开始测试 W&B 连接")
    
    # 测量登录时间
    log_with_timestamp("🔍 尝试登录 W&B...")
    login_start_time = time.time()
    try:
        wandb.login()
        login_duration = time.time() - login_start_time
        log_with_timestamp(f"✅ W&B 登录成功，耗时：{login_duration:.2f} 秒")
    except wandb.errors.AuthenticationError as e:
        logger.error(f"❌ W&B 登录失败：{e}")
        return
    except Exception as e:
        logger.error(f"❌ 未知错误导致登录失败：{e}")
        return

    # 测量 W&B 初始化和同步时间
    log_with_timestamp("🚀 正在初始化 W&B 运行...")
    init_start_time = time.time()
    try:
        # run = wandb.init(project="test_connection", job_type="connectivity_test")
        run = wandb.init(project="test_connection", job_type="connectivity_test", settings=wandb.Settings(disable_git=True))
        init_duration = time.time() - init_start_time
        log_with_timestamp(f"✅ 成功初始化 W&B 运行，耗时：{init_duration:.2f} 秒")

        # 测量数据同步时间
        log_with_timestamp("📤 开始数据同步...")
        sync_start_time = time.time()
        run.log({"connectivity_check": "success"})
        run.finish()
        sync_duration = time.time() - sync_start_time
        log_with_timestamp(f"✅ 数据同步完成，耗时：{sync_duration:.2f} 秒")
    except wandb.errors.CommError as e:
        logger.error(f"❌ 网络连接失败：{e}")
    except wandb.errors.UsageError as e:
        logger.error(f"❌ 初始化失败，可能是项目设置错误：{e}")
    except Exception as e:
        logger.error(f"❌ 无法初始化 W&B 运行：{e}")
        return

    total_duration = time.time() - init_start_time
    log_with_timestamp(f"🕒 总耗时：{total_duration:.2f} 秒")

if __name__ == "__main__":
    measure_network_latency()
    test_wandb_connectivity()