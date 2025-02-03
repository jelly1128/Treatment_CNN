import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
import torchvision.transforms as transforms

from config.config import load_config
from models import  MultiLabelDetectionModel
from evaluate import ModelEvaluator
from utils.torch_utils import get_device_and_num_gpus, set_seed
from data.dataloader import create_train_val_dataloaders
from data.datasets import MultiLabelDetectionDataset

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

    
def setup_data(config):
    train_data_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=360),
        transforms.ToTensor(),
        ])
    
    val_data_transforms = transforms.Compose([
        transforms.ToTensor(),
        ])
    
    train_datasets_list = []
    for split in [NOW_FOLD[0], NOW_FOLD[1]]:
        dataset = MultiLabelDetectionDataset(config.paths.root,
                                             transform=train_data_transforms,
                                             num_classes=config.training.num_classes,
                                             split=split)
        train_datasets_list.append(dataset)
    train_datasets = ConcatDataset(train_datasets_list)
    val_datasets = MultiLabelDetectionDataset(config.paths.root,
                                             transform=val_data_transforms,
                                             num_classes=config.training.num_classes,
                                             split=NOW_FOLD[2])
    
    return train_datasets, val_datasets

def plot_dataset_samples(dataloader):
    # サンプルを表示し、1つの画像として保存
    num_samples_to_show = 10
    fig, axes = plt.subplots(1, num_samples_to_show, figsize=(20, 4))

    for i, (images, img_names, labels) in enumerate(dataloader):
        if i >= 1:  # 1バッチだけ処理
            break
        for j in range(num_samples_to_show):
            ax = axes[j]
            img = images[j].permute(1, 2, 0).numpy()  # CHW to HWC, tensor to numpy
            ax.imshow(img)
            ax.set_title(f"Label: {labels[j]}")
            ax.axis('off')
            print(f"Image path: {img_names[j]}, Label: {labels[j]}")

    plt.tight_layout()
    plt.savefig('dataset_samples.png')
    
def show_dataset_stats(dataloader):
    # データセットの総数
    total_samples = len(dataloader.dataset)
    
    # ラベルの分布を計算
    all_labels = []
    for batch, (images, _, labels) in enumerate(dataloader):
        all_labels.extend(labels.cpu().tolist())
    
    # クラスごとのサンプル数をカウントするためのカウンターを初期化
    class_samples = Counter()

    # One-hotラベルを処理
    for one_hot in all_labels:
        # one_hotがリストであることを確認
        if isinstance(one_hot, list):
            one_hot = torch.tensor(one_hot)  # リストをテンソルに変換
        # 1の位置を見つけてカウントを更新
        for idx, value in enumerate(one_hot):
            if value == 1:
                class_samples[idx] += 1

    print(f"総サンプル数: {total_samples}")
    print("クラスごとのサンプル数:")
    for class_label, count in sorted(class_samples.items()):
        print(f"クラス {class_label}: {count}")

def train_val(config):
    # setup
    device, num_gpus = get_device_and_num_gpus()
    set_seed(42)
    train_datasets, val_datasets = setup_data(config)
    
    train_dataloader, val_dataloaders = create_train_val_dataloaders(config, NOW_FOLD, num_gpus)
    
    print(len(train_datasets))
    print(len(val_datasets))
    
    # debug
    plot_dataset_samples(train_dataloader)
    # show_dataset_stats(train_dataloader)
    os._exit(0)
    
    model = MultiLabelDetectionModel(num_classes=config.training.num_classes,
                                     pretrained=config.training.pretrained,
                                     freeze_backbone=config.training.freeze_backbone)
    
    if num_gpus > 1:
        model = nn.DataParallel(model)  # モデルを複数GPUで学習する場合
    model = model.to(device)
    
    print(f'Training {config.training.mode} model')
    
    # 学習パラメータ
    optimizer = optim.Adam(model.parameters(), lr=float(config.training.learning_rate))
    criterion = nn.BCEWithLogitsLoss()
    
    # 学習の経過を保存
    loss_history = {'train': [], 'val': []}
    best_validation_loss = float('inf')
    
    evaluator = ModelEvaluator(config.paths.save_name)
    
    for epoch in range(config.training.max_epoch_num):
        training_loss = 0.0
        validation_loss = 0.0
        
        # training
        model.train()
        
        for batch_idx, (images, _, labels) in enumerate(train_dataloader):  # batch_idxとimageを取得
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images) # BCEWithLogitsLoss を使用している場合，モデルの出力はlogits 確率じゃない！
            # print(outputs.size())

            loss = criterion(outputs, labels)
            training_loss += loss.item()

            loss.backward()
            optimizer.step()
        
        avg_training_loss = training_loss / len(train_dataloader)
        loss_history['train'].append(avg_training_loss)
        
        # validation
        model.eval()

        with torch.no_grad():
            for batch_idx, (images, _, labels) in enumerate(val_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)  # ラベルと比較
                
                validation_loss += loss.item()
        
        avg_validation_loss = validation_loss / len(val_dataloader)
        loss_history['val'].append(avg_validation_loss)

        if best_validation_loss > avg_validation_loss:
            best_validation_loss = avg_validation_loss
            torch.save(model.state_dict(), f"{config.paths.save_name}/model_best.pth")
            saved_str = " ==> model saved"
        else:
            saved_str = ""
            
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'{config.paths.save_name}/epoch_{epoch+1}_model.pth')

        print("epoch %d: training_loss:%.4f validation_loss:%.4f %s" %
              (epoch + 1, avg_training_loss, avg_validation_loss, saved_str))
            
    evaluator.plot_learning_curve(loss_history)
    evaluator.save_loss_to_csv(loss_history)
    
def parse_args():
    parser = argparse.ArgumentParser(description='Multi-class organ classification model using ResNet')
    parser.add_argument('-c', '--config', help='Path to config file', default='config.yaml')
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', type=float)
    parser.add_argument('-b', '--batch_size', help='Batch size', type=int)
    parser.add_argument('-m', '--max_epoch_num', help='Maximum number of training epochs', type=int)
    return parser.parse_args()
    
def main():
    args = parse_args()
    config = load_config(args.config)

    # Command line arguments override config file
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.max_epoch_num is not None:
        config.training.max_epoch_num = args.max_epoch_num
        
    # 結果保存folderを作成
    if not os.path.exists(os.path.join(config.paths.save_dir)):
        os.makedirs(config.paths.save_dir, exist_ok=True)
    
    # Use the config in your code
    train_val(config)

if __name__ == '__main__':
    main()