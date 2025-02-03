import yaml
from pathlib import Path
from .schema import Config, TrainingConfig, PathConfig

def load_config(config_path: Path) -> Config:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        
    training_config = TrainingConfig(**config_dict['training'])
    path_config = PathConfig(**config_dict['paths'])
    
    return Config(training=training_config, paths=path_config)