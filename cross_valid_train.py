import logging
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from config.config import load_train_config
from data.dataloader import create_multilabel_train_dataloaders
from data.visualization import plot_dataset_samples, show_dataset_stats
from engine.trainer import Trainer
from engine.validator import Validator
from model.setup_models import setup_model
from utils.torch_utils import get_device_and_num_gpus, set_seed
from utils.logger import setup_logging
from utils.evaluator import ModelEvaluator


SPLIT1 = (
    "20210119093456_000001-001",
    # "20210531112330_000005-001",
    # "20211223090943_000001-002",
    # "20230718-102254-ES06_20230718-102749-es06-hd",
    # "20230802-104559-ES09_20230802-105630-es09-hd",
)

SPLIT2 = (
    "20210119093456_000001-002",
    # "20210629091641_000001-002",
    # "20211223090943_000001-003",
    # "20230801-125025-ES06_20230801-125615-es06-hd",
    # "20230803-110626-ES06_20230803-111315-es06-hd"
)

SPLIT3 = (
    "20210119093456_000002-001",
    # "20210630102301_000001-002",
    # "20220322102354_000001-002",
    # "20230802-095553-ES09_20230802-101030-es09-hd",
    # "20230803-093923-ES09_20230803-094927-es09-hd",
)

SPLIT4 = (
    "20210524100043_000001-001",
    # "20210531112330_000001-001",
    # "20211021093634_000001-001",
    # "20211021093634_000001-003"
)

FOLD1 = (SPLIT1, SPLIT2, SPLIT3)
FOLD2 = (SPLIT2, SPLIT3, SPLIT4)
FOLD3 = (SPLIT3, SPLIT4, SPLIT1)
FOLD4 = (SPLIT4, SPLIT1, SPLIT2)

NOW_FOLD = FOLD1


def train_val(config, fold):
    # setup
    device, num_gpus = get_device_and_num_gpus()
    set_seed(42)
    train_dataloader, val_dataloader = create_multilabel_train_dataloaders(config, fold, num_gpus)

    # debug
    # plot_dataset_samples(config.paths.save_dir, train_dataloader)
    # show_dataset_stats(train_dataloader)

    model = setup_model(config, device, num_gpus, mode='train')

    # 学習パラメータ
    optimizer = optim.Adam(model.parameters(), lr=float(config.training.learning_rate))
    criterion = nn.BCEWithLogitsLoss()

    # 学習の経過を保存
    loss_history = {'train': [], 'val': []}
    evaluator = ModelEvaluator(config.paths.save_dir)

    # 学習・検証エンジン
    trainer = Trainer(model, optimizer, criterion, device)
    validator = Validator(model, criterion, device)
    
    # os._exit(0)

    # 学習ループ
    for epoch in range(config.training.max_epochs):
        train_loss = trainer.train_epoch(train_dataloader)
        val_loss = validator.validate(val_dataloader)

        loss_history['train'].append(train_loss)
        loss_history['val'].append(val_loss)

        # ログ出力
        log_message = "epoch %d: training_loss: %.4f validation_loss: %.4f" % (
            epoch + 1, train_loss, val_loss)
        
        logging.info(log_message)

        # モデル保存
        if val_loss <= min(loss_history['val']):
            torch.save(model.state_dict(), os.path.join(config.paths.save_dir, "best_model.pth"))

    # 学習曲線の可視化
    evaluator.plot_learning_curve(loss_history)
    evaluator.save_loss_to_csv(loss_history)
    
def parse_args():
    parser = argparse.ArgumentParser(description='Multi-class organ classification model using ResNet')
    parser.add_argument('-c', '--config', help='Path to config file', default='config.yaml')
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', type=float)
    parser.add_argument('-b', '--batch_size', help='Batch size', type=int)
    parser.add_argument('-m', '--max_epochs', help='Maximum number of training epochs', type=int)
    return parser.parse_args()
    
def main():
    args = parse_args()
    config = load_train_config(args.config)
    
    # 結果保存フォルダを作成
    os.makedirs(config.paths.save_dir, exist_ok=True)
    
    setup_logging(config.paths.save_dir)

    # Command line arguments override config file
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.max_epochs is not None:
        config.training.max_epochs = args.max_epochs
        
    # Use the config in your code
    train_val(config, NOW_FOLD)

if __name__ == '__main__':
    main()