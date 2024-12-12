import wandb
import logging
import time
import socket
import os
import sys
import requests
import json
import psutil
from datetime import datetime

# 设置详细的日志记录
# yanwen 到此一游

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def measure_network_latency():
    """测量所有WandB相关endpoint的网络延迟"""
    endpoints = [
        'https://api.wandb.ai',
        'https://wandb.ai',
        'https://files.wandb.ai'
    ]
    
    results = {}
    for endpoint in endpoints:
        try:
            start = time.time()
            response = requests.get(f"{endpoint}/health", timeout=10)
            latency = time.time() - start
            results[endpoint] = {
                'latency': latency,
                'status': response.status_code
            }
            logger.info(f"Endpoint {endpoint}: Latency = {latency*1000:.2f}ms, Status = {response.status_code}")
        except Exception as e:
            results[endpoint] = {'error': str(e)}
            logger.error(f"Failed to connect to {endpoint}: {str(e)}")
    return results

def check_system_resources():
    """检查系统资源使用情况"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    net_io = psutil.net_io_counters()
    
    resources = {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'disk_percent': disk.percent,
        'net_bytes_sent': net_io.bytes_sent,
        'net_bytes_recv': net_io.bytes_recv
    }
    
    logger.info(f"System Resources: {json.dumps(resources, indent=2)}")
    return resources

def check_environment():
    """检查环境变量和系统信息"""
    logger.info("Checking environment...")
    env_vars = {
        'WANDB_API_KEY': os.environ.get('WANDB_API_KEY', 'Not set'),
        'WANDB_MODE': os.environ.get('WANDB_MODE', 'Not set'),
        'HTTPS_PROXY': os.environ.get('HTTPS_PROXY', 'Not set'),
        'HTTP_PROXY': os.environ.get('HTTP_PROXY', 'Not set'),
        'WANDB_DEBUG': os.environ.get('WANDB_DEBUG', 'Not set'),
        'PYTHONPATH': os.environ.get('PYTHONPATH', 'Not set'),
    }
    
    if env_vars['WANDB_API_KEY'] != 'Not set':
        env_vars['WANDB_API_KEY'] = 'Present but hidden'
    
    logger.info(f"Environment variables: {json.dumps(env_vars, indent=2)}")
    return env_vars

def test_wandb_connection():
    """完整的WandB连接测试和性能分析"""
    hostname = socket.gethostname()
    logger.info(f"Starting comprehensive WandB test from host: {hostname}")
    
    try:
        # 1. 环境检查
        env_vars = check_environment()
        
        # 2. 系统资源基线检查
        logger.info("Checking system resources before initialization...")
        resources_before = check_system_resources()
        
        # 3. 网络延迟测试
        logger.info("Measuring network latency...")
        network_latency = measure_network_latency()
        
        # 4. 登录测试
        logger.info("Testing WandB login...")
        login_start = time.time()
        login_status = wandb.login()
        login_time = time.time() - login_start
        logger.info(f"Login completed in {login_time:.2f} seconds. Status: {login_status}")
        
        # 5. 初始化测试
        logger.info("Starting WandB initialization...")
        init_start = time.time()
        
        run = wandb.init(
            project="test-project",
            entity="yichao-utaustin",
            name=f"diagnostic-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            settings=wandb.Settings(
                init_timeout=300,
                _disable_stats=True,
                _disable_meta=True
            ),
            mode='offline'  # 使用离线模式确保测试可靠性
        )
        
        init_time = time.time() - init_start
        logger.info(f"WandB initialization took {init_time:.2f} seconds")
        
        # 6. 测试日志记录
        logger.info("Testing logging capability...")
        log_start = time.time()
        wandb.log({"test_metric": 1.0})
        log_time = time.time() - log_start
        
        # 7. 检查初始化后的系统资源
        resources_after = check_system_resources()
        
        # 8. 收集完整结果
        results = {
            'hostname': hostname,
            'environment': env_vars,
            'network_latency': network_latency,
            'timing': {
                'login': login_time,
                'init': init_time,
                'log': log_time,
                'total': time.time() - login_start
            },
            'resources': {
                'before': resources_before,
                'after': resources_after,
                'delta': {
                    k: resources_after[k] - resources_before[k]
                    for k in resources_before
                    if isinstance(resources_before[k], (int, float))
                }
            }
        }
        
        # 9. 保存诊断结果
        results_file = f'wandb_diagnostic_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Diagnostic results saved to {results_file}")
        
        # 10. 清理
        wandb.finish()
        logger.info("WandB test completed successfully!")
        return True, results
        
    except Exception as e:
        logger.error(f"WandB test failed with error: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        return False, {'error': str(e)}

if __name__ == "__main__":
    # 设置调试选项
    os.environ['WANDB_DEBUG'] = 'true'
    os.environ['WANDB_CONSOLE'] = 'wrap'
    
    try:
        success, diagnostic_results = test_wandb_connection()
        if not success:
            logger.error("Diagnostic test failed!")
            sys.exit(1)
        
        # 打印关键性能指标
        timing = diagnostic_results.get('timing', {})
        logger.info("\nPerformance Summary:")
        logger.info(f"Login Time: {timing.get('login', 'N/A')}s")
        logger.info(f"Init Time: {timing.get('init', 'N/A')}s")
        logger.info(f"Log Time: {timing.get('log', 'N/A')}s")
        logger.info(f"Total Time: {timing.get('total', 'N/A')}s")
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        sys.exit(1)