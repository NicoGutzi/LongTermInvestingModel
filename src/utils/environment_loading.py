import os
import yaml
import logging

def load_config(config_path: str = None) -> dict:
    """
    Load configuration from a YAML file.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            '..',
            'configs',
            'config.yaml'
        )
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logging.info(f"Configuration loaded from {config_path}")
            return config
    except Exception as e:
        logging.error(f"Failed to load configuration file: {e}")
        return {}
