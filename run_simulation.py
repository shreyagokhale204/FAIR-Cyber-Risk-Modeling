
import sys

from config.model_config import get_config, validate_config

print("Python interpreter:", sys.executable)

config = get_config()
validate_config()

print("NovaFlow FAIR configuration loaded successfully!")
