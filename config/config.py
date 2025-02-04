import yaml
from pathlib import Path
from .schema import Config, TrainingConfig, TestConfig, PathConfig

def load_train_config(config_path: Path) -> Config:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        
    training_config = TrainingConfig(**config_dict['training'])
    path_config = PathConfig(**config_dict['paths'])
    
    return Config(training=training_config, paths=path_config)

def load_test_config(config_path: Path) -> Config:
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)
        
    test_config = TestConfig(**config_data['test'])
    path_config = PathConfig(**config_data['paths'])
    
    return Config(test=test_config, paths=path_config)
