import logging
from pathlib import Path

def setup_logging(save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(save_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )