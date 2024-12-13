# mamba_config_help.py
import sys
from mamba_ssm.models.config_mamba import MambaConfig

def print_help():
    """Print the help information for MambaConfig."""
    try:
        print("===== Help Information for MambaConfig =====")
        help(MambaConfig)
        print("============================================")
    except Exception as e:
        print(f"Error: Unable to retrieve help for MambaConfig. {e}")
        sys.exit(1)

if __name__ == "__main__":
    print_help()
