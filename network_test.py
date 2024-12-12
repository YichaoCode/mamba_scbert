import socket
import ssl
import requests
import logging
import subprocess
from urllib.parse import urlparse

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_connection(url):
    """测试给定URL的各种连接方式"""
    parsed = urlparse(url)
    hostname = parsed.netloc
    port = 443 if parsed.scheme == 'https' else 80
    
    logger.info(f"\nTesting connection to {url}")
    
    # 1. 测试DNS解析
    try:
        ip = socket.gethostbyname(hostname)
        logger.info(f"DNS Resolution: {hostname} -> {ip}")
    except socket.gaierror as e:
        logger.error(f"DNS Resolution failed: {e}")
        return
    
    # 2. 测试TCP连接
    try:
        sock = socket.create_connection((hostname, port), timeout=10)
        logger.info(f"TCP Connection to port {port}: Success")
        sock.close()
    except Exception as e:
        logger.error(f"TCP Connection failed: {e}")
        return
        
    # 3. 测试SSL连接
    try:
        context = ssl.create_default_context()
        with socket.create_connection((hostname, port)) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                logger.info(f"SSL Connection: Success")
                logger.info(f"SSL Version: {ssock.version()}")
                logger.info(f"Cipher: {ssock.cipher()}")
    except Exception as e:
        logger.error(f"SSL Connection failed: {e}")
    
    # 4. 测试HTTP/HTTPS请求
    try:
        response = requests.get(url, timeout=10)
        logger.info(f"HTTP/HTTPS Request: {response.status_code}")
        logger.info(f"Response Headers: {dict(response.headers)}")
    except Exception as e:
        logger.error(f"HTTP/HTTPS Request failed: {e}")
    
    # 5. 测试curl（如果可用）
    try:
        result = subprocess.run(['curl', '-v', url], capture_output=True, text=True)
        logger.debug(f"Curl output:\n{result.stderr}")
    except Exception as e:
        logger.error(f"Curl test failed: {e}")

if __name__ == "__main__":
    urls = [
        'https://api.wandb.ai/health',
        'https://wandb.ai/health',
        'https://files.wandb.ai/health'
    ]
    
    for url in urls:
        test_connection(url)