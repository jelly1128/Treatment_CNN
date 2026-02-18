from pathlib import Path

import yaml

from config.schema import (
    ExperimentConfig,
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    ExperimentPaths,
    CVSplitsConfig,
)

def load_experiment_config(config_path: Path) -> ExperimentConfig:
    """
    YAMLファイルから実験の設定を読み込む関数。

    Args:
        config_path (Path): 設定ファイルのパス。

    Returns:
        ExperimentConfig: 読み込まれた実験の設定を保持するデータクラスのインスタンス。
    """
    if not config_path.exists():
        raise FileNotFoundError(f"指定された設定ファイルが見つかりません: {config_path}")
    
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    # YAML設定ファイルの構造の確認
    required_keys = {'mode', 'model', 'dataset', 'paths', 'cv_splits'}
    if not required_keys.issubset(config_dict.keys()):
        raise ValueError(f"設定ファイルの構造が正しくありません: {config_dict}")
    
    return ExperimentConfig(
        mode=config_dict['mode'],
        model=ModelConfig(**config_dict['model']),
        dataset=DatasetConfig(**config_dict['dataset']),
        paths=ExperimentPaths(**config_dict['paths']),
        cv_splits=CVSplitsConfig(config_dict['cv_splits']),
        training=TrainingConfig(**config_dict['training']) if 'training' in config_dict else None
    )