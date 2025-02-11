import logging
from pathlib import Path
import time

# 日本の時刻に設定
def jst_time(*args):
    return time.localtime(time.time() + 9 * 3600)  # UTC+9

logging.Formatter.converter = jst_time


def setup_logging(save_dir: str, mode='training') -> logging.Logger:
    """
    ログの設定を実施。
    
    引数:
        save_dir (str): ログを保存するディレクトリのパス
        mode (str, optional): ログの種類を指定します。デフォルトは 'training'
        
    戻り値:
        logger (logging.Logger): ログ
    """
    # ログを保存するディレクトリ
    save_dir = Path(save_dir)
    
    if not save_dir.exists():
        raise FileNotFoundError(f"ログの保存ディレクトリ '{save_dir}' が存在しません．")
    
    logger = logging.getLogger(mode)
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # ログフォーマット
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # ファイル出力のハンドラー
    file_handler = logging.FileHandler(save_dir / f'{mode}.log', encoding="utf-8")
    file_handler.setFormatter(formatter)

    # コンソール出力のハンドラー
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # ロガーにハンドラーを追加
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger
