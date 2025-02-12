import logging
import argparse
from pathlib import Path
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim

from config.config_loader import load_train_config
from data.data_splitter import CrossValidationSplitter
from data.dataloader import DataLoaderFactory, create_multilabel_train_dataloaders
from data.visualization import plot_dataset_samples, show_dataset_stats
from engine.trainer import Trainer
from engine.validator import Validator
from model.setup_models import setup_model
from utils.torch_utils import get_device_and_num_gpus, set_seed
from utils.logger import setup_logging
from utils.evaluator import ModelEvaluator


# SPLIT1 = (
#     "20210119093456_000001-001",
#     "20210531112330_000005-001",
#     "20211223090943_000001-002",
#     "20230718-102254-ES06_20230718-102749-es06-hd",
#     "20230802-104559-ES09_20230802-105630-es09-hd",
# )

# SPLIT2 = (
#     "20210119093456_000001-002",
#     "20210629091641_000001-002",
#     "20211223090943_000001-003",
#     "20230801-125025-ES06_20230801-125615-es06-hd",
#     "20230803-110626-ES06_20230803-111315-es06-hd"
# )

# SPLIT3 = (
#     "20210119093456_000002-001",
#     "20210630102301_000001-002",
#     "20220322102354_000001-002",
#     "20230802-095553-ES09_20230802-101030-es09-hd",
#     "20230803-093923-ES09_20230803-094927-es09-hd",
# )

# SPLIT4 = (
#     "20210524100043_000001-001",
#     "20210531112330_000001-001",
#     "20211021093634_000001-001",
#     "20211021093634_000001-003"
# )

# FOLD1 = (SPLIT1, SPLIT2, SPLIT3)
# FOLD2 = (SPLIT2, SPLIT3, SPLIT4)
# FOLD3 = (SPLIT3, SPLIT4, SPLIT1)
# FOLD4 = (SPLIT4, SPLIT1, SPLIT2)

# NOW_FOLD = FOLD1

def train_val(config):
    # setup
    device, num_gpus = get_device_and_num_gpus()
    set_seed(42)
    
    # dataloaderの作成
    ### 要検討 ###
    # train_dataloader, val_dataloader = create_multilabel_train_dataloaders(config, fold, num_gpus)
    
    splitter = CrossValidationSplitter(splits=config.splits.root)
    split_folders = splitter.get_split_folders()
    fold_splits = splitter.get_fold_splits()
    
    # debug
    for _, (folders, splits) in enumerate(zip(split_folders, fold_splits)):
        print(f"Train splits: {splits['train']}")
        print(f"Val split: {splits['val']}")
        print(f"Test split: {splits['test']}")
        print(f"Number of train folders: {len(folders['train'])}")
        print(folders['train'])
        print(f"Number of val folders: {len(folders['val'])}")
        print(folders['val'])
        print(f"Number of val folders: {len(folders['test'])}")
        print(folders['test'])
    
    # 理想
    train_data_dirs = [split_folders[0]['train']]
    val_data_dirs = [split_folders[0]['train']]
    dataloader_factory = DataLoaderFactory(
        dataset_root=config.paths.dataset_root,
        batch_size=config.training.batch_size,
        num_classes=config.training.num_classes,
        num_gpus=num_gpus
    )
    train_dataloader = dataloader_factory.create_multilabel_dataloaders(train_data_dirs)
    val_dataloader = dataloader_factory.create_multilabel_dataloaders(val_data_dirs)
    
    os._exit(0)

    # debug
    plot_dataset_samples(config.paths.save_dir, train_dataloader)
    show_dataset_stats(train_dataloader)

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
    parser = argparse.ArgumentParser(description='Multi-class treatment classification model using ResNet')
    parser.add_argument('-c', '--config', help='Path to config file', default='config.yaml')
    return parser.parse_args()
    
def main():
    # 設定読み込み
    args = parse_args()
    config = load_train_config(args.config)
    
    ### 要検討 ###
    # folds = generate_folds(config.splits.root)
    
    # 結果保存フォルダを作成
    # Path(config.paths.save_dir).mkdir(exist_ok=True)
    
    # for fold_name, fold in folds.items():
    #     fold_path = Path(config.paths.save_dir, fold_name)
    #     # 結果保存フォルダを作成
    #     fold_path.mkdir(exist_ok=True)
    
    # ログ設定
    setup_logging(config.paths.save_dir, mode='training')
    
    # debug_fold = list(config.splits.root.values())[0]
    # debug_fold = config.splits.root
    # print(debug_fold)
    # print(type(debug_fold))
    train_val(config)
    
    
    # train_val(config, NOW_FOLD)
    
    # train_val(config, NOW_FOLD)
    # for fold_idx, fold in enumerate((FOLD1, FOLD2, FOLD3, FOLD4)):
    #     fold_path = os.path.join(config.paths.save_dir, f'fold{fold_idx+1}')
    #     # 結果保存フォルダを作成
    #     os.makedirs(fold_path, exist_ok=True)
    #     config.paths.save_dir = fold_path
    #     train_val(config, fold)

if __name__ == '__main__':
    main()