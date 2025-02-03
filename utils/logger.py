import logging
from pathlib import Path

def setup_logging(save_dir: str):
    """ロギングの初期化"""
    save_dir = Path(save_dir)  # strをPathに変換
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(save_dir / "training.log"),  # Pathオブジェクトを使用
            logging.StreamHandler(),
        ],
    )