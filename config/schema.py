from enum import Enum
from pathlib import Path
import torch.nn as nn

from typing import Literal
from pydantic import BaseModel, RootModel


# ──────────────────────────────────────────
# Enum 定義
# ──────────────────────────────────────────

class ModelArchitecture(str, Enum):
    """
    使用可能なモデルのアーキテクチャを表す列挙型
    
    追加方法：
    1. 新しいモデルアーキテクチャをこの列挙型に追加
    2. cnn_models.py 内の MODEL_ARCHITECTURES 辞書に新しいモデルのクラスを追加
    """
    RESNET18 = 'resnet18'
    RESNET34 = 'resnet34'
    RESNET50 = 'resnet50'


class ModelType(str, Enum):
    """
    モデルのタイプを表す列挙型
    
    追加方法：
    1. 新しいモデルタイプをこの列挙型に追加
    2. cnn_models.py 内の MODEL_TYPES 辞書に新しいモデルタイプのクラスを追加
    """
    MULTITASK    = "multitask"     # マルチラベル分類（BCEWithLogitsLoss）
    SINGLE_LABEL = "single_label"  # シングルラベル分類（CrossEntropyLoss）

    @property
    def criterion(self):
        """モデルタイプに対応する損失関数を返す"""
        match self:
            case ModelType.MULTITASK:
                return nn.BCEWithLogitsLoss()
            case ModelType.SINGLE_LABEL:
                return nn.CrossEntropyLoss()

# ──────────────────────────────────────────
# サブ設定クラス
# ──────────────────────────────────────────
class ModelConfig(BaseModel):
    """train / test で共通のモデル設定"""
    architecture: ModelArchitecture
    type: ModelType
    num_classes: int

class DatasetConfig(BaseModel):
    """データセットに関する設定を保持するデータクラス"""
    root: str
    img_size: int
    batch_size: int

class TrainingConfig(BaseModel):
    """モデルのトレーニングに関する設定を保持するデータクラス"""
    pretrained: bool
    freeze_backbone: bool
    learning_rate: float
    max_epochs: int

class ExperimentPaths(BaseModel):
    """
    入出力パスに関する設定を保持するデータクラス
    """
    save_dir: str
    model_paths: list[str] | None = None   # 学習済みモデルのパスリスト（テスト用）
    
class CVSplitsConfig(RootModel[dict[str, list[str]]]):
    """
    交差検証の分割設定．

    YAML 構造:
        splits:
          split1: [video_dir_a, video_dir_b, ...]
          split2: [video_dir_c, ...]
    """
    pass


# ──────────────────────────────────────────
# 全体の設定クラス
# ──────────────────────────────────────────

class ExperimentConfig(BaseModel):
    """
    実験全体の設定を保持するデータクラス
    """
    mode: Literal['train', 'test']
    model: ModelConfig
    dataset: DatasetConfig
    paths: ExperimentPaths
    cv_splits: CVSplitsConfig
    training: TrainingConfig | None = None  # トレーニング設定（テストモードでは不要）