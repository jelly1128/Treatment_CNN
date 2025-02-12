# data/data_utils.py
import os
import matplotlib.pyplot as plt
from collections import Counter

import torch
from torch.utils.data import ConcatDataset, DataLoader
import torchvision.transforms as transforms

from datasets.multilabel_dataset import MultiLabelDetectionDataset

# ここでは、クロスバリデーション用の分割例として定義（必要に応じて調整）
SPLIT1 = ("20210119093456_000001-001",)
SPLIT2 = ("20210119093456_000001-002",)
SPLIT3 = ("20210119093456_000002-001",)
SPLIT4 = ("20210524100043_000001-001",)

# クロスバリデーション用の Fold 定義
FOLD1 = (SPLIT1, SPLIT2, SPLIT3)
# ※ 他の Fold も同様に定義可能

# 今回は FOLD1 を使用する例
NOW_FOLD = FOLD1

def setup_data(config):
    """データセットの生成と前処理の定義"""
    train_data_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=360),
        transforms.ToTensor(),
    ])
    
    val_data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_datasets_list = []
    # ここでは Fold の前半 2 分割を訓練データに使用
    for split in [NOW_FOLD[0], NOW_FOLD[1]]:
        dataset = MultiLabelDetectionDataset(dataset_root=config.paths.dataset_root,
                                             transform=train_data_transforms,
                                             num_classes=config.training.num_classes,
                                             split=split)
        train_datasets_list.append(dataset)
    train_datasets = ConcatDataset(train_datasets_list)
    
    # 残りの分割を検証用データとする
    val_datasets = MultiLabelDetectionDataset(dataset_root=config.paths.dataset_root,
                                               transform=val_data_transforms,
                                               num_classes=config.training.num_classes,
                                               split=NOW_FOLD[2])
    
    return train_datasets, val_datasets

def plot_dataset_samples(dataloader, save_path='dataset_samples.png'):
    """データローダからサンプル画像を1バッチ分取得し、画像一覧をプロットして保存する"""
    num_samples_to_show = 10
    fig, axes = plt.subplots(1, num_samples_to_show, figsize=(20, 4))

    for i, (images, img_names, labels) in enumerate(dataloader):
        if i >= 1:  # 1バッチのみ処理
            break
        for j in range(num_samples_to_show):
            ax = axes[j]
            # tensor: CHW -> HWC に変換し、numpy 配列に
            img = images[j].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(f"Label: {labels[j]}")
            ax.axis('off')
            print(f"Image path: {img_names[j]}, Label: {labels[j]}")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved dataset samples to {save_path}")

def show_dataset_stats(dataloader):
    """データセットのサンプル数やクラス分布を表示する"""
    total_samples = len(dataloader.dataset)
    
    all_labels = []
    for _, (_, _, labels) in enumerate(dataloader):
        all_labels.extend(labels.cpu().tolist())
    
    class_samples = Counter()
    for one_hot in all_labels:
        # one_hot がリストの場合はテンソルに変換
        if isinstance(one_hot, list):
            one_hot = torch.tensor(one_hot)
        for idx, value in enumerate(one_hot):
            if value == 1:
                class_samples[idx] += 1

    print(f"Total samples: {total_samples}")
    print("Class distribution:")
    for class_label, count in sorted(class_samples.items()):
        print(f"Class {class_label}: {count}")
