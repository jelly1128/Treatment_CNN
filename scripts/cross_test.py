import os
import yaml
from dataclasses import dataclass
from typing import Optional
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import csv
from PIL import Image, ImageDraw

from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from data_transformer import DataTransformer
from datasets import AnomalyDetectionDatasetForTest, TreatmentClassificationDatasetForTest, MultiLabelDetectionDatasetForTest
from models import AnomalyDetectionModel, TreatmentClassificationModel, MultiLabelDetectionModel
from evaluate_afafa import ModelEvaluator

from config.config_loader import load_test_config
from data.dataloader import create_multi_label_test_dataloaders
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

TEST_SPLIT = SPLIT3

        


        
# ROC曲線の描画、Youden's Indexでの閾値探索、分類精度の計算
def plot_roc_curve(y_true, y_prob, class_idx, save_dir):
    # ROC曲線の計算
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Youden's Indexの計算
    youdens_index = tpr - fpr
    best_threshold_idx = np.argmax(youdens_index)  # Youden's Indexが最大となるインデックス
    best_threshold = thresholds[best_threshold_idx]  # 最適な閾値
    best_fpr = fpr[best_threshold_idx]
    best_tpr = tpr[best_threshold_idx]
    
    # ROC曲線の描画
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    # Youden's Indexの最適な点を描画
    plt.scatter(best_fpr, best_tpr, color='red', marker='o', label=f'Youden\'s Index (Best Threshold = {best_threshold:.2f})')
    
    # グラフの装飾
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class {class_idx}')
    plt.legend(loc="lower right")

    # 画像の保存
    roc_image_path = os.path.join(save_dir, f'roc_curve_class_{class_idx}.png')
    plt.savefig(roc_image_path)
    plt.close()  # 画像を閉じる


def find_best_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_index = tpr - fpr
    best_threshold_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_threshold_idx]
    return best_threshold


def evaluate_classification(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return accuracy, precision, recall, f1


def test_anomaly_detection_model(config):
    # setup
    device, num_gpus = get_device_and_num_gpus()
    set_seed(42)
    train_dataloader, val_dataloader = create_train_val_dataloaders(config, fold, num_gpus)
    
    model = MultiLabelDetectionModel(num_classes=config.test.num_classes)
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # モデルの読み込み
    state_dict = torch.load(config.paths.model)
    if 'module.' in list(state_dict.keys())[0]:
        # state_dictがDataParallelで保存されている場合
        if num_gpus > 1:
            model.load_state_dict(state_dict)
        else:
            # DataParallelなしでモデルを読み込む場合、'module.'プレフィックスを削除
            model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    else:
        # state_dictがDataParallelなしで保存されている場合
        if num_gpus > 1:
            # DataParallelを使用する場合、'module.'プレフィックスを追加
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    
            
    # 学習の経過を保存
    model.eval()
    
    # すべての結果を保存
    all_probabilities = []
    all_predictions = []
    all_ground_truths = []
    all_image_paths = []
    
    # GradCAMのターゲットを定義
    # target_layers = [model.module.resnet.layer4[-1]]
    # GradCAMの計算
    # cam = GradCAM(model=model, target_layers=target_layers)
    
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    for folder_name in TEST_SPLIT:
        # 各テストフォルダ毎の結果の保存folderを作成
        if not os.path.exists(os.path.join(config.paths.save_dir, folder_name)):
            os.mkdir(os.path.join(config.paths.save_dir, folder_name))
        
        test_dataset = MultiLabelDetectionDatasetForTest(os.path.join(config.paths.dataset_root), folder_name, data_transform, config.test.num_classes)
        
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4 * num_gpus)
        
        # debug
        # plot_dataset_samples(test_dataloader)
        # show_dataset_stats(test_dataloader)
        print(folder_name)
        
        subfolder_probabilities = []
        subfolder_labels = []
        subfolder_image_paths = []
        
        # grad-camなし
        with torch.no_grad():
            for images, image_paths, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                # outputsの各要素に対してsigmoidを適用し、それを各クラスごとに出力する
                sigmoid_outputs = torch.sigmoid(outputs[0])

                # 出力が多次元の場合、ネストされたリストに対してフォーマットを適用する
                # formatted_outputs = [f"{prob:.4f}" for prob in sigmoid_outputs.tolist()]
                formatted_outputs = [round(prob, 4) for prob in sigmoid_outputs.tolist()]
                
                # print(image_paths[0], labels[0].cpu().numpy(), formatted_outputs)
                
                subfolder_probabilities.append(formatted_outputs)
                subfolder_labels.append(labels[0].cpu().numpy())
                subfolder_image_paths.extend(image_paths)
                
                # print(subfolder_probabilities[0], subfolder_labels[0], subfolder_image_paths[0])
                # print(subfolder_labels[0])
                # os._exit(0)
                
        # CSVファイルへの保存設定
        output_csv_file = f'{os.path.join(config.paths.save_dir, folder_name)}/multi_labels_test_with_labels.csv'
        with open(output_csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # ヘッダーを定義（クラス数に応じた列名）
            header = ['Image_Path'] + [f"Class_{i}_Prob" for i in range(config.test.num_classes)] + [f"Class_{i}_Label" for i in range(config.test.num_classes)]
            writer.writerow(header)
            
            # 各画像のパスとそれに対応する確率・ラベルを1行ずつ書く
            for image_path, probabilities, labels in zip(subfolder_image_paths, subfolder_probabilities, subfolder_labels):
                writer.writerow([image_path] + probabilities + labels.tolist())  # ラベルをリストとして展開
                
        print("make csvfile")
        
        # 予測ラベルを生成（閾値50%）
        pred_labels = (np.array(subfolder_probabilities) >= 0.5).astype(int)
        
        # 予測確率が全てのクラスで50%未満の画像を処理
        for i, probabilities in enumerate(subfolder_probabilities):
            if all(prob < 0.5 for prob in probabilities):  # すべてのクラスで50%未満の場合
                max_prob_index = np.argmax(probabilities)  # 最大の確率を持つクラスのインデックスを取得
                pred_labels[i][max_prob_index] = 1         # 最大確率のクラスのラベルを1に設定
        
        with open(f'{os.path.join(config.paths.save_dir, folder_name)}/multi_labels_test_with_labels_50%.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # ヘッダーを定義
            header = ['Image_Path'] + [f"Pred_Class_{i}" for i in range(config.test.num_classes)] + [f"True_Class_{i}" for i in range(config.test.num_classes)]
            writer.writerow(header)
            
            # 予測ラベルと正解ラベルを1行ずつ書く
            for image_path, pred, true in zip(subfolder_image_paths, pred_labels, subfolder_labels):
                writer.writerow([image_path] + pred.tolist() + true.tolist())
                
        # 精度、適合率、再現率、F1スコアを計算し、CSVファイルに保存
        threshold_50_metrics_file = f'{os.path.join(config.paths.save_dir, folder_name)}/classification_metrics_50%.csv'
        conf_matrix_50_file = f'{os.path.join(config.paths.save_dir, folder_name)}/confusion_matrices_50%.csv'

        with open(threshold_50_metrics_file, mode='w', newline='') as metrics_file, \
             open(conf_matrix_50_file, mode='w', newline='') as conf_matrix_file:
            
            metrics_writer = csv.writer(metrics_file)
            conf_matrix_writer = csv.writer(conf_matrix_file)

            # メトリクスファイルのヘッダー
            metrics_writer.writerow(['Class', 'Accuracy', 'Precision', 'Recall', 'F1_Score'])
            
            # 混同行列ファイルのヘッダー
            conf_matrix_writer.writerow(['Class', 'TP', 'FP', 'TN', 'FN'])

            # 各クラスごとに計算
            for class_idx in range(config.test.num_classes):
                true_labels = [label[class_idx] for label in subfolder_labels]  # 該当クラスの真のラベル
                pred_class_labels = pred_labels[:, class_idx]  # 50%閾値での予測ラベル
                
                # print(true_labels)
                # print(pred_class_labels)
                
                # 精度、適合率、再現率、F1スコアを計算
                accuracy = accuracy_score(true_labels, pred_class_labels)
                precision = precision_score(true_labels, pred_class_labels, zero_division=0)
                recall = recall_score(true_labels, pred_class_labels, zero_division=0)
                f1 = f1_score(true_labels, pred_class_labels, zero_division=0)

                # メトリクスをCSVファイルに書き込み
                metrics_writer.writerow([class_idx, f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])

                # 混同行列の計算 (TN, FP, FN, TP の順)
                conf_matrix = confusion_matrix(true_labels, pred_class_labels, labels=[0, 1])
                tn, fp, fn, tp = conf_matrix.ravel()  # 混同行列の要素を展開

                # 混同行列をCSVに保存（50%閾値の結果）
                conf_matrix_writer.writerow([class_idx, tp, fp, tn, fn])

                print(f'Class {class_idx} - Confusion Matrix (50% Threshold)')
                print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

                # コンソール出力
                print(f"Class {class_idx} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
                
        # サブフォルダごとの処理の最後にタイムラインの可視化を追加
        # DataFrameの生成
        columns_predicted = [f"Predicted_Class_{i}" for i in range(config.test.num_classes)]
        columns_labels = [f"Label_Class_{i}" for i in range(config.test.num_classes)]
        df = pd.DataFrame(
            data=np.hstack([pred_labels, np.array(subfolder_labels)]),
            columns=columns_predicted + columns_labels
        )
        df['Image_Path'] = subfolder_image_paths  # 画像パスを追加

        # タイムラインの可視化
        visualize_multi_label_timeline(
            df=df,
            save_dir=os.path.join(config.paths.save_dir, folder_name),
            filename="predicted",
            num_classes=config.test.num_classes
        )

        visualize_ground_truth_timeline(
            df=df,
            save_dir=os.path.join(config.paths.save_dir, folder_name),
            filename="ground_truth"
        )

        print(f"Visualization for folder {folder_name} completed.")
 

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-class organ classification model using ResNet')
    parser.add_argument('-c', '--config', help='Path to config file', default='config.yaml')
    return parser.parse_args()
    
def main():
    args = parse_args()
    config = load_test_config(args.config)
    setup_logging(config.paths.save_dir)
    
    # 結果保存フォルダを作成
    os.makedirs(config.paths.save_dir, exist_ok=True)
    
    os._exit(0)
    
    tester = Tester(config, device)

    for folder_name in TEST_SPLIT:
        test_dataloader = create_test_dataloader(config, folder_name, num_gpus)
        tester.test(test_dataloader, folder_name)

if __name__ == '__main__':
    main()