import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from config.config_loader import load_train_config
from data.data_splitter import CrossValidationSplitter
from data.dataloader import DataLoaderFactory
from utils.torch_utils import get_device_and_num_gpus, set_seed
from utils.logger import setup_logging
from utils.training_monitor import TrainingMonitor
from engine.trainer import Trainer
from engine.validator import Validator
from model.setup_models import setup_model


def train_val(config: dict, train_data_dirs: list, val_data_dirs: list, fold_idx: int):
    logging.info(f"Starting training for fold {fold_idx+1}...")

    # setup
    device, num_gpus = get_device_and_num_gpus()
    set_seed(42)

    # データローダーの作成
    dataloader_factory = DataLoaderFactory(
        dataset_root=config.paths.dataset_root,
        batch_size=config.training.batch_size,
        num_classes=config.training.num_classes,
        num_gpus=num_gpus
    )
    train_dataloader, val_dataloader = dataloader_factory.create_multilabel_dataloaders(train_data_dirs, val_data_dirs)

    # モデルのセットアップ
    model = setup_model(config, device, num_gpus, mode='train')

    # 学習パラメータ
    optimizer = optim.Adam(model.parameters(), lr=float(config.training.learning_rate))
    criterion = nn.BCEWithLogitsLoss()

    # 学習の経過を保存
    loss_history = {'train': [], 'val': []}
    monitor = TrainingMonitor(Path(config.paths.save_dir) / f"fold_{fold_idx+1}")

    # 学習・検証エンジン
    trainer = Trainer(model, optimizer, criterion, device)
    validator = Validator(model, criterion, device)

    # 学習ループ
    for epoch in range(config.training.max_epochs):
        train_loss = trainer.train_epoch(train_dataloader)
        val_loss = validator.validate(val_dataloader)

        loss_history['train'].append(train_loss)
        loss_history['val'].append(val_loss)

        # ログ出力
        logging.info(
            f"- Epoch {epoch+1} Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n"
        )

        # モデル保存
        if val_loss <= min(loss_history['val']):
            model_path = Path(config.paths.save_dir) / f"fold_{fold_idx+1}" / "best_model.pth"
            torch.save(model.state_dict(), model_path)
            logging.info(f"Best model for fold {fold_idx+1} saved.")

    # 学習曲線の可視化
    monitor.plot_learning_curve(loss_history)
    monitor.save_loss_to_csv(loss_history)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Path to config file', default='config.yaml')
    parser.add_argument('-f', '--fold', type=int, help='Fold number to train (optional, if not set, trains all folds)')
    return parser.parse_args()


def main():
    # 設定読み込み
    args = parse_args()
    config = load_train_config(args.config)

    # 結果保存フォルダを作成
    Path(config.paths.save_dir).mkdir(exist_ok=True)

    # ログ設定
    setup_logging(config.paths.save_dir, mode='training')

    # 交差検証のためのデータ分割
    splitter = CrossValidationSplitter(splits=config.splits.root)
    split_folders = splitter.get_split_folders()

    # debug
    # splits = splitter.get_fold_splits()
    # print(splits)
    # import sys
    # sys.exit()

    # --fold 引数が指定された場合は、そのfoldのみ学習
    if args.fold is not None:
        if args.fold < 0 or args.fold >= len(split_folders):
            raise ValueError(f"指定されたfold {args.fold} は無効です (0〜{len(split_folders)-1} の範囲で指定してください)")
        
        train_data_dirs = split_folders[args.fold]['train']
        val_data_dirs = split_folders[args.fold]['val']
        train_val(config, train_data_dirs, val_data_dirs, args.fold)
    
    # すべてのfoldを学習
    else:
        for fold_idx, fold in enumerate(split_folders):
            train_data_dirs = fold['train']
            val_data_dirs = fold['val']
            train_val(config, train_data_dirs, val_data_dirs, fold_idx)


if __name__ == '__main__':
    main()
