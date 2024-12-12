# test_mamba_config.py
import logging
import inspect
from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

# 设置日志
logging.basicConfig(level=logging.INFO)

def test_mamba_config():
    # 1. 打印 MambaConfig 的构造函数签名
    logging.info("MambaConfig signature:")
    logging.info(inspect.signature(MambaConfig))
    
    # 2. 打印所有可用属性
    logging.info("\nAvailable attributes in MambaConfig:")
    logging.info(dir(MambaConfig))
    
    # 3. 创建一个默认配置实例
    config = MambaConfig()
    
    # 4. 打印默认配置的所有属性
    logging.info("\nDefault config values:")
    for attr in dir(config):
        if not attr.startswith('_'):  # 只打印非内部属性
            try:
                value = getattr(config, attr)
                logging.info(f"{attr}: {value}")
            except Exception as e:
                logging.info(f"{attr}: Error getting value - {e}")
    
    # 5. 尝试修改一些配置
    try:
        # 创建一个新配置实例并设置一些值
        test_config = MambaConfig()
        test_config.d_model = 16
        test_config.n_layer = 4
        test_config.vocab_size = 1000
        
        # 尝试创建模型
        model = MambaLMHeadModel(test_config)
        logging.info("\nSuccessfully created model with test config")
        logging.info(f"Model structure:\n{model}")
        
    except Exception as e:
        logging.error(f"\nError during test configuration: {e}")

if __name__ == "__main__":
    test_mamba_config()