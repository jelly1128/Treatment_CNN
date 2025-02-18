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
from data.dataset_visualizer import plot_dataset_samples, show_dataset_stats



def train_val(config: dict, train_data_dirs: list, val_data_dirs: list):
    # setup
    device, num_gpus = get_device_and_num_gpus()
    set_seed(42)
    
    dataloader_factory = DataLoaderFactory(
        dataset_root=config.paths.dataset_root,
        batch_size=config.training.batch_size,
        num_classes=config.training.num_classes,
        num_gpus=num_gpus
    )
    
    train_dataloader, val_dataloader = dataloader_factory.create_multilabel_dataloaders(train_data_dirs, val_data_dirs)
    
    # visualize
    # plot_dataset_samples(config.paths.save_dir, train_dataloader)
    # show_dataset_stats(train_dataloader)
    # show_dataset_stats(val_dataloader)
    
    model = setup_model(config, device, num_gpus, mode='train')
    
    # 学習パラメータ
    optimizer = optim.Adam(model.parameters(), lr=float(config.training.learning_rate))
    criterion = nn.BCEWithLogitsLoss()

    # 学習の経過を保存
    loss_history = {'train': [], 'val': []}
    monitor = TrainingMonitor(config.paths.save_dir)

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
        log_message = "epoch %d: training_loss: %.4f validation_loss: %.4f" % (
                       epoch+1,  train_loss,         val_loss)
        
        logging.info(log_message)

        # モデル保存
        if val_loss <= min(loss_history['val']):
            torch.save(model.state_dict(), Path(config.paths.save_dir, "best_model.pth"))
            logging.info("Best model saved.")

    # 学習曲線の可視化
    monitor.plot_learning_curve(loss_history)
    monitor.save_loss_to_csv(loss_history)
    
def parse_args():
    parser = argparse.ArgumentParser(description='Multi-class treatment classification model using ResNet')
    parser.add_argument('-c', '--config', help='Path to config file', default='config.yaml')
    return parser.parse_args()
    
def main():
    # 設定読み込み
    args = parse_args()
    config = load_train_config(args.config)
    
    # 結果保存フォルダを作成
    Path(config.paths.save_dir).mkdir(exist_ok=True)
    
    # ログ設定
    setup_logging(config.paths.save_dir, mode='training')
    
    # データ分割
    # 各foldのtrain/val/test用フォルダ名リストを取得
    splitter = CrossValidationSplitter(splits=config.splits.root)
    split_folders = splitter.get_split_folders()
    print(split_folders[1]['train'])
    print(split_folders[1]['val'])
    
    # debug
    # fold_idx=0のtrainとvalのデータディレクトリを取得
    train_data_dirs = split_folders[1]['train']
    val_data_dirs = split_folders[1]['val']
    train_val(config, train_data_dirs, val_data_dirs)

if __name__ == '__main__':
    main()