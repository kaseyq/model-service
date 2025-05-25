# src/common/config_utils.py
from src.common.file_utils import load_yaml
from src.common.path_utils import PathUtil

def load_config():
    return load_yaml([PathUtil.get_path("config.yml"), PathUtil.get_path("config.yaml")])
