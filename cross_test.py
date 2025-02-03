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
from evaluate import ModelEvaluator


SPLIT1 = (
    "20210119093456_000001-001",
    "20210531112330_000005-001",
    "20211223090943_000001-002",
    "20230718-102254-ES06_20230718-102749-es06-hd",
    "20230802-104559-ES09_20230802-105630-es09-hd",
)

SPLIT2 = (
    "20210119093456_000001-002",
    "20210629091641_000001-002",
    "20211223090943_000001-003",
    "20230801-125025-ES06_20230801-125615-es06-hd",
    "20230803-110626-ES06_20230803-111315-es06-hd"
)

SPLIT3 = (
    "20210119093456_000002-001",
    "20210630102301_000001-002",
    "20220322102354_000001-002",
    "20230802-095553-ES09_20230802-101030-es09-hd",
    "20230803-093923-ES09_20230803-094927-es09-hd",
)

SPLIT4 = (
    "20210524100043_000001-001",
    "20210531112330_000001-001",
    "20211021093634_000001-001",
    "20211021093634_000001-003"
)

TEST_SPLIT = SPLIT3


@dataclass
class TestConfig:
    mode: str
    img_size: int
    n_class: int
    # 動的に追加できるように属性を追加
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

@dataclass
class PathConfig:
    root: str
    model: str
    save_name: str
    # 動的に追加できるように属性を追加
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

@dataclass
class Config:
    test: TestConfig
    paths: PathConfig

def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    test_config = TestConfig(**config_dict['test'])
    path_config = PathConfig(**config_dict['paths'])
    
    return Config(test=test_config, paths=path_config)

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPU{'s' if num_gpus > 1 else ''}")
    else:
        device = torch.device("cpu")
        num_gpus = 0
        print("Using CPU")
    
    print("Device being used:", device)
    return device, num_gpus

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

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
    
    label_distribution = Counter(all_labels)
    
    # クラスごとのサンプル数
    class_samples = dict(label_distribution)
    
    print(f"総サンプル数: {total_samples}")
    print("クラスごとのサンプル数:")
    for class_label, count in sorted(class_samples.items()):
        print(f"クラス {class_label}: {count}")
        
def visualize_dataset(dataset, output_dir, num_samples=500):
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(0, len(dataset), 100):
        images_list = []
        labels_list = []
        
        for j in range(i, min(i + 100, len(dataset))):
            images, image_path, label = dataset[j]
            last_image = images[-1]
            images_list.append(last_image)
            labels_list.append(label)
        
        grid = make_grid(torch.stack(images_list), nrow=10)
        
        # 画像をPIL Imageに変換
        pil_image = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype('uint8'))
        
        # 画像にラベルを追加
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.load_default()
        label_text = ", ".join(map(str, set(labels_list)))
        draw.text((10, 10), f"Labels: {label_text}", fill=(255, 255, 255), font=font)
        
        # 画像を保存
        output_path = os.path.join(output_dir, f"sample_{i//100}.png")
        pil_image.save(output_path)
        
        print(f"Saved image with labels {label_text} to {output_path}")
        
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
    device, num_gpus = setup_device()
    setup_seed(42)
    
    model = MultiLabelDetectionModel(n_class=config.test.n_class)
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
        if not os.path.exists(os.path.join(config.paths.save_name, folder_name)):
            os.mkdir(os.path.join(config.paths.save_name, folder_name))
        
        test_dataset = MultiLabelDetectionDatasetForTest(os.path.join(config.paths.root), folder_name, data_transform, config.test.n_class)
        
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
        output_csv_file = f'{os.path.join(config.paths.save_name, folder_name)}/multilabels_test_with_labels.csv'
        with open(output_csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # ヘッダーを定義（クラス数に応じた列名）
            header = ['Image_Path'] + [f"Class_{i}_Prob" for i in range(config.test.n_class)] + [f"Class_{i}_Label" for i in range(config.test.n_class)]
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
        
        with open(f'{os.path.join(config.paths.save_name, folder_name)}/multilabels_test_with_labels_50%.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # ヘッダーを定義
            header = ['Image_Path'] + [f"Pred_Class_{i}" for i in range(config.test.n_class)] + [f"True_Class_{i}" for i in range(config.test.n_class)]
            writer.writerow(header)
            
            # 予測ラベルと正解ラベルを1行ずつ書く
            for image_path, pred, true in zip(subfolder_image_paths, pred_labels, subfolder_labels):
                writer.writerow([image_path] + pred.tolist() + true.tolist())
                
        # 精度、適合率、再現率、F1スコアを計算し、CSVファイルに保存
        threshold_50_metrics_file = f'{os.path.join(config.paths.save_name, folder_name)}/classification_metrics_50%.csv'
        conf_matrix_50_file = f'{os.path.join(config.paths.save_name, folder_name)}/confusion_matrices_50%.csv'

        with open(threshold_50_metrics_file, mode='w', newline='') as metrics_file, \
             open(conf_matrix_50_file, mode='w', newline='') as conf_matrix_file:
            
            metrics_writer = csv.writer(metrics_file)
            conf_matrix_writer = csv.writer(conf_matrix_file)

            # メトリクスファイルのヘッダー
            metrics_writer.writerow(['Class', 'Accuracy', 'Precision', 'Recall', 'F1_Score'])
            
            # 混同行列ファイルのヘッダー
            conf_matrix_writer.writerow(['Class', 'TP', 'FP', 'TN', 'FN'])

            # 各クラスごとに計算
            for class_idx in range(config.test.n_class):
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
        columns_predicted = [f"Predicted_Class_{i}" for i in range(config.test.n_class)]
        columns_labels = [f"Label_Class_{i}" for i in range(config.test.n_class)]
        df = pd.DataFrame(
            data=np.hstack([pred_labels, np.array(subfolder_labels)]),
            columns=columns_predicted + columns_labels
        )
        df['Image_Path'] = subfolder_image_paths  # 画像パスを追加
        
        def visualize_multilabel_timeline(df, save_dir, filename, n_class):
            # Define the colors for each class
            label_colors = {
                0: (254, 195, 195),       # white
                1: (204, 66, 38),         # lugol
                2: (57, 103, 177),        # indigo
                3: (96, 165, 53),         # nbi
                4: (86, 65, 72),          # custom color for label 4
                5: (159, 190, 183),       # custom color for label 5
            }

            # Default color for labels not specified in label_colors
            default_color = (148, 148, 148)

            # Extract the predicted labels columns
            predicted_labels = df[[col for col in df.columns if 'Predicted' in col]].values

            # Determine the number of images
            n_images = len(predicted_labels)
            
            # Set timeline height based on the number of labels
            timeline_width = n_images
            timeline_height = n_class * (n_images // 10)

            # Create a blank image for the timeline
            timeline_image = Image.new('RGB', (timeline_width, timeline_height), (255, 255, 255))
            draw = ImageDraw.Draw(timeline_image)

            # Iterate over each image (row in the CSV)
            for i in range(n_images):
                # Get the predicted labels for the current image
                labels = predicted_labels[i]
                
                # Check each label and draw corresponding rectangles
                for label_idx, label_value in enumerate(labels):
                    if label_value == 1:
                        row_idx = label_idx

                        # Calculate the position in the timeline
                        x1 = i * (timeline_width // n_images)
                        x2 = (i + 1) * (timeline_width // n_images)
                        y1 = row_idx * (n_images // 10)
                        y2 = (row_idx + 1) * (n_images // 10)
                        
                        # Get the color for the current label
                        color = label_colors.get(label_idx, default_color)
                        
                        # Draw the rectangle for the label
                        draw.rectangle([x1, y1, x2, y2], fill=color)
                        
            # Save the image
            timeline_image.save(os.path.join(save_dir, f'{filename}_multilabel_timeline.png'))
            print(f'Timeline image saved at {os.path.join(save_dir, "multilabel_timeline.png")}')
            
            
        def visualize_ground_truth_timeline(df, save_dir, filename):
            # Define the colors for each class
            label_colors = {
                0: (254, 195, 195),       # white
                1: (204, 66, 38),         # lugol
                2: (57, 103, 177),        # indigo
                3: (96, 165, 53),         # nbi
                4: (86, 65, 72),          # custom color for label 4
                5: (159, 190, 183),       # custom color for label 5
            }

            # Default color for labels not specified in label_colors
            default_color = (148, 148, 148)

            # Extract the ground truth label columns
            ground_truth_labels = df[[col for col in df.columns if 'Label' in col]].values
            
            # Debug: Check if Label_Class_5 exists and has non-zero values
            # print(f"Label_Class_5 exists in DataFrame: {'Label_Class_5' in df.columns}")
            # print(f"Label_Class_5 value counts:\n{df['Label_Class_5'].value_counts()}") if 'Label_Class_5' in df.columns else None
            
            # Determine the number of images
            n_images = len(ground_truth_labels)

            # Set timeline dimensions
            timeline_width = n_images
            timeline_height = 2 * (n_images // 10)  # 20 pixels per label row (2 rows total)

            # Create a blank image for the timeline
            timeline_image = Image.new('RGB', (timeline_width, timeline_height), (255, 255, 255))
            draw = ImageDraw.Draw(timeline_image)

            # Iterate over each image (row in the CSV)
            for i in range(n_images):
                # Get the ground truth labels for the current image
                labels = ground_truth_labels[i]
                
                # Check each label and draw corresponding rectangles
                for label_idx, label_value in enumerate(labels):
                    if label_value == 1:
                        # Determine the correct row for drawing
                        row_idx = 0 if label_idx < 6 else 1

                        # Calculate the position in the timeline
                        x1 = i * (timeline_width // n_images)
                        x2 = (i + 1) * (timeline_width // n_images)
                        y1 = row_idx * (n_images // 10)  # Each row is 20 pixels tall
                        y2 = (row_idx + 1) * (n_images // 10)  # Height for the rectangle
                        
                        # Get the color for the current label
                        color = label_colors.get(label_idx, default_color)
                        
                        # Draw the rectangle for the label
                        draw.rectangle([x1, y1, x2, y2], fill=color)

            # Save the image
            timeline_image.save(os.path.join(save_dir, f'{filename}_ground_truth_timeline.png'))
            print(f'Ground truth timeline image saved at {os.path.join(save_dir, f"{filename}_ground_truth_timeline.png")}')

        # タイムラインの可視化
        visualize_multilabel_timeline(
            df=df,
            save_dir=os.path.join(config.paths.save_name, folder_name),
            filename="predicted",
            n_class=config.test.n_class
        )

        visualize_ground_truth_timeline(
            df=df,
            save_dir=os.path.join(config.paths.save_name, folder_name),
            filename="ground_truth"
        )

        print(f"Visualization for folder {folder_name} completed.")
        
    # for maker in ['olympus', 'fujifilm']:
    #     subfolder_names = [name for name in os.listdir(os.path.join(config.paths.root, maker)) if os.path.isdir(os.path.join(config.paths.root, maker, name))]
    #     print(subfolder_names)
    #     data_transform = DataTransformer.get_transform(maker, 'test', config.test.img_size)
        
    #     for subfolder_name in subfolder_names:
    #         # 各テストフォルダ毎の結果の保存folderを作成
    #         if not os.path.exists(os.path.join(config.paths.save_name, subfolder_name)):
    #             os.mkdir(os.path.join(config.paths.save_name, subfolder_name))
                
    #         test_dataset = MultiLabelDetectionDatasetForTest(os.path.join(config.paths.root, maker), 
    #                                                       subfolder_name, 
    #                                                       data_transform,
    #                                                       config.test.n_class)
    #                                                     #   15)
    #         test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4 * num_gpus)
                        
    #         # debug
    #         # plot_dataset_samples(test_dataloader)
    #         # show_dataset_stats(test_dataloader)
    #         print(subfolder_name)
    #         # continue
    #         # os._exit(0)
            
    #         subfolder_probabilities = []
    #         subfolder_labels = []
    #         subfolder_image_paths = []
            
    #         # grad-camなし
    #         with torch.no_grad():
    #             for images, image_paths, labels in test_dataloader:
    #                 images = images.to(device)
    #                 labels = labels.to(device)

    #                 outputs = model(images)
    #                 # outputsの各要素に対してsigmoidを適用し、それを各クラスごとに出力する
    #                 sigmoid_outputs = torch.sigmoid(outputs[0])

    #                 # 出力が多次元の場合、ネストされたリストに対してフォーマットを適用する
    #                 # formatted_outputs = [f"{prob:.4f}" for prob in sigmoid_outputs.tolist()]
    #                 formatted_outputs = [round(prob, 4) for prob in sigmoid_outputs.tolist()]
                    
    #                 # print(image_paths[0], labels[0].cpu().numpy(), formatted_outputs)
                    
    #                 subfolder_probabilities.append(formatted_outputs)
    #                 subfolder_labels.append(labels[0].cpu().numpy())
    #                 subfolder_image_paths.extend(image_paths)
                    
    #                 # print(subfolder_probabilities[0], subfolder_labels[0], subfolder_image_paths[0])
    #                 # print(subfolder_labels[0])
    #                 # os._exit(0)
                    
    #         # CSVファイルへの保存設定
    #         output_csv_file = f'{os.path.join(config.paths.save_name, subfolder_name)}/multilabels_test_with_labels.csv'
    #         with open(output_csv_file, mode='w', newline='') as file:
    #             writer = csv.writer(file)
                
    #             # ヘッダーを定義（クラス数に応じた列名）
    #             header = ['Image_Path'] + [f"Class_{i}_Prob" for i in range(config.test.n_class)] + [f"Class_{i}_Label" for i in range(config.test.n_class)]
    #             writer.writerow(header)
                
    #             # 各画像のパスとそれに対応する確率・ラベルを1行ずつ書く
    #             for image_path, probabilities, labels in zip(subfolder_image_paths, subfolder_probabilities, subfolder_labels):
    #                 writer.writerow([image_path] + probabilities + labels.tolist())  # ラベルをリストとして展開
                    
    #         print("make csvfile")
                    
            # os._exit(0)
                    
            # 各クラスのROC曲線、Youden's Indexによる最適閾値計算、分類精度を出力・CSVに保存
            # metrics_csv_file = f'{os.path.join(config.paths.save_name, subfolder_name)}/classification_metrics.csv'
            # conf_matrix_csv_file = f'{os.path.join(config.paths.save_name, subfolder_name)}/confusion_matrices.csv'

            # with open(metrics_csv_file, mode='w', newline='') as metrics_file, \
            #     open(conf_matrix_csv_file, mode='w', newline='') as conf_matrix_file:

            #     metrics_writer = csv.writer(metrics_file)
            #     conf_matrix_writer = csv.writer(conf_matrix_file)

            #     # メトリクスファイルのヘッダー
            #     metrics_header = ['Class', 'Best_Threshold', 'Accuracy', 'Precision', 'Recall', 'F1_Score']
            #     metrics_writer.writerow(metrics_header)

            #     # 混同行列ファイルのヘッダー
            #     conf_matrix_writer.writerow(['Class', 'TP', 'FP', 'TN', 'FN'])

            #     # 保存用のディレクトリ作成
            #     save_dir = os.path.join(config.paths.save_name, subfolder_name)
            #     os.makedirs(save_dir, exist_ok=True)

            #     best_thresholds = []
                
            #     for class_idx in range(config.test.n_class):
            #         true_labels = [label[class_idx] for label in subfolder_labels]  # 該当クラスの真のラベル
            #         pred_probs = [float(prob[class_idx]) for prob in subfolder_probabilities]  # 該当クラスの予測確率

            #         # ROC曲線を描画し画像として保存
            #         plot_roc_curve(true_labels, pred_probs, class_idx, save_dir)

            #         # Youden's Index による最適閾値の計算
            #         best_threshold = find_best_threshold(true_labels, pred_probs)
            #         best_thresholds.append(best_threshold)

            #         # 最適閾値を用いて予測を 0, 1 に分類
            #         pred_labels = (np.array(pred_probs) >= best_threshold).astype(int)

            #         # 分類精度を計算
            #         accuracy, precision, recall, f1 = evaluate_classification(true_labels, pred_probs, best_threshold)

            #         # メトリクスをCSVに書き込む
            #         metrics_writer.writerow([class_idx, f"{best_threshold:.4f}", f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])

            #         # 混同行列の計算 (TN, FP, FN, TP の順)
            #         conf_matrix = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
            #         tn, fp, fn, tp = conf_matrix.ravel()  # 混同行列の要素を展開

            #         # 混同行列をCSVに保存
            #         conf_matrix_writer.writerow([class_idx, tp, fp, tn, fn])

            #         print(f'Class {class_idx} - Confusion Matrix')
            #         print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

            #         # コンソール出力
            #         print(f'Class {class_idx} - Best threshold: {best_threshold:.4f}')
            #         print(f"Class {class_idx} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
            
            # # 予測ラベルを生成（閾値50%）
            # pred_labels = (np.array(subfolder_probabilities) >= 0.5).astype(int)
            
            # with open(f'{os.path.join(config.paths.save_name, subfolder_name)}/multilabels_test_with_labels_50%.csv', mode='w', newline='') as file:
            #     writer = csv.writer(file)
                
            #     # ヘッダーを定義
            #     header = ['Image_Path'] + [f"Pred_Class_{i}" for i in range(config.test.n_class)] + [f"True_Class_{i}" for i in range(config.test.n_class)]
            #     writer.writerow(header)
                
            #     # 予測ラベルと正解ラベルを1行ずつ書く
            #     for image_path, pred, true in zip(subfolder_image_paths, pred_labels, subfolder_labels):
            #         writer.writerow([image_path] + pred.tolist() + true.tolist())
                    
            # # 精度、適合率、再現率、F1スコアを計算し、CSVファイルに保存
            # threshold_50_metrics_file = f'{os.path.join(config.paths.save_name, subfolder_name)}/classification_metrics_50%.csv'
            # conf_matrix_50_file = f'{os.path.join(config.paths.save_name, subfolder_name)}/confusion_matrices_50%.csv'

            # with open(threshold_50_metrics_file, mode='w', newline='') as metrics_file, \
            #      open(conf_matrix_50_file, mode='w', newline='') as conf_matrix_file:
                
            #     metrics_writer = csv.writer(metrics_file)
            #     conf_matrix_writer = csv.writer(conf_matrix_file)

            #     # メトリクスファイルのヘッダー
            #     metrics_writer.writerow(['Class', 'Accuracy', 'Precision', 'Recall', 'F1_Score'])
                
            #     # 混同行列ファイルのヘッダー
            #     conf_matrix_writer.writerow(['Class', 'TP', 'FP', 'TN', 'FN'])

            #     # 各クラスごとに計算
            #     for class_idx in range(config.test.n_class):
            #         true_labels = [label[class_idx] for label in subfolder_labels]  # 該当クラスの真のラベル
            #         pred_class_labels = pred_labels[:, class_idx]  # 50%閾値での予測ラベル
                    
            #         # print(true_labels)
            #         # print(pred_class_labels)
                    
            #         # 精度、適合率、再現率、F1スコアを計算
            #         accuracy = accuracy_score(true_labels, pred_class_labels)
            #         precision = precision_score(true_labels, pred_class_labels, zero_division=0)
            #         recall = recall_score(true_labels, pred_class_labels, zero_division=0)
            #         f1 = f1_score(true_labels, pred_class_labels, zero_division=0)

            #         # メトリクスをCSVファイルに書き込み
            #         metrics_writer.writerow([class_idx, f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])

            #         # 混同行列の計算 (TN, FP, FN, TP の順)
            #         conf_matrix = confusion_matrix(true_labels, pred_class_labels, labels=[0, 1])
            #         tn, fp, fn, tp = conf_matrix.ravel()  # 混同行列の要素を展開

            #         # 混同行列をCSVに保存（50%閾値の結果）
            #         conf_matrix_writer.writerow([class_idx, tp, fp, tn, fn])

            #         print(f'Class {class_idx} - Confusion Matrix (50% Threshold)')
            #         print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

            #         # コンソール出力
            #         print(f"Class {class_idx} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
                    
            # # サブフォルダごとの処理の最後にタイムラインの可視化を追加
            # # DataFrameの生成
            # columns_predicted = [f"Predicted_Class_{i}" for i in range(config.test.n_class)]
            # columns_labels = [f"Label_Class_{i}" for i in range(config.test.n_class)]
            # df = pd.DataFrame(
            #     data=np.hstack([pred_labels, np.array(subfolder_labels)]),
            #     columns=columns_predicted + columns_labels
            # )
            # df['Image_Path'] = subfolder_image_paths  # 画像パスを追加
            
            # def visualize_multilabel_timeline(df, save_dir, filename, n_class):
            #     # Define the colors for each class
            #     label_colors = {
            #         0: (254, 195, 195),       # white
            #         1: (204, 66, 38),         # lugol
            #         2: (57, 103, 177),        # indigo
            #         3: (96, 165, 53),         # nbi
            #         4: (86, 65, 72),          # custom color for label 4
            #         5: (159, 190, 183),       # custom color for label 5
            #     }

            #     # Default color for labels not specified in label_colors
            #     default_color = (148, 148, 148)

            #     # Extract the predicted labels columns
            #     predicted_labels = df[[col for col in df.columns if 'Predicted' in col]].values

            #     # Determine the number of images
            #     n_images = len(predicted_labels)
                
            #     # Set timeline height based on the number of labels
            #     timeline_width = n_images
            #     timeline_height = n_class * (n_images // 10)

            #     # Create a blank image for the timeline
            #     timeline_image = Image.new('RGB', (timeline_width, timeline_height), (255, 255, 255))
            #     draw = ImageDraw.Draw(timeline_image)

            #     # Iterate over each image (row in the CSV)
            #     for i in range(n_images):
            #         # Get the predicted labels for the current image
            #         labels = predicted_labels[i]
                    
            #         # Check each label and draw corresponding rectangles
            #         for label_idx, label_value in enumerate(labels):
            #             if label_value == 1:
            #                 row_idx = label_idx

            #                 # Calculate the position in the timeline
            #                 x1 = i * (timeline_width // n_images)
            #                 x2 = (i + 1) * (timeline_width // n_images)
            #                 y1 = row_idx * (n_images // 10)
            #                 y2 = (row_idx + 1) * (n_images // 10)
                            
            #                 # Get the color for the current label
            #                 color = label_colors.get(label_idx, default_color)
                            
            #                 # Draw the rectangle for the label
            #                 draw.rectangle([x1, y1, x2, y2], fill=color)
                            
            #     # Save the image
            #     timeline_image.save(os.path.join(save_dir, f'{filename}_multilabel_timeline.png'))
            #     print(f'Timeline image saved at {os.path.join(save_dir, "multilabel_timeline.png")}')
                
                
            # def visualize_ground_truth_timeline(df, save_dir, filename):
            #     # Define the colors for each class
            #     label_colors = {
            #         0: (254, 195, 195),       # white
            #         1: (204, 66, 38),         # lugol
            #         2: (57, 103, 177),        # indigo
            #         3: (96, 165, 53),         # nbi
            #         4: (86, 65, 72),          # custom color for label 4
            #         5: (159, 190, 183),       # custom color for label 5
            #     }

            #     # Default color for labels not specified in label_colors
            #     default_color = (148, 148, 148)

            #     # Extract the ground truth label columns
            #     ground_truth_labels = df[[col for col in df.columns if 'Label' in col]].values
                
            #     # Debug: Check if Label_Class_5 exists and has non-zero values
            #     # print(f"Label_Class_5 exists in DataFrame: {'Label_Class_5' in df.columns}")
            #     # print(f"Label_Class_5 value counts:\n{df['Label_Class_5'].value_counts()}") if 'Label_Class_5' in df.columns else None
                
            #     # Determine the number of images
            #     n_images = len(ground_truth_labels)

            #     # Set timeline dimensions
            #     timeline_width = n_images
            #     timeline_height = 2 * (n_images // 10)  # 20 pixels per label row (2 rows total)

            #     # Create a blank image for the timeline
            #     timeline_image = Image.new('RGB', (timeline_width, timeline_height), (255, 255, 255))
            #     draw = ImageDraw.Draw(timeline_image)

            #     # Iterate over each image (row in the CSV)
            #     for i in range(n_images):
            #         # Get the ground truth labels for the current image
            #         labels = ground_truth_labels[i]
                    
            #         # Check each label and draw corresponding rectangles
            #         for label_idx, label_value in enumerate(labels):
            #             if label_value == 1:
            #                 # Determine the correct row for drawing
            #                 row_idx = 0 if label_idx < 6 else 1

            #                 # Calculate the position in the timeline
            #                 x1 = i * (timeline_width // n_images)
            #                 x2 = (i + 1) * (timeline_width // n_images)
            #                 y1 = row_idx * (n_images // 10)  # Each row is 20 pixels tall
            #                 y2 = (row_idx + 1) * (n_images // 10)  # Height for the rectangle
                            
            #                 # Get the color for the current label
            #                 color = label_colors.get(label_idx, default_color)
                            
            #                 # Draw the rectangle for the label
            #                 draw.rectangle([x1, y1, x2, y2], fill=color)

            #     # Save the image
            #     timeline_image.save(os.path.join(save_dir, f'{filename}_ground_truth_timeline.png'))
            #     print(f'Ground truth timeline image saved at {os.path.join(save_dir, f"{filename}_ground_truth_timeline.png")}')

            # # タイムラインの可視化
            # visualize_multilabel_timeline(
            #     df=df,
            #     save_dir=os.path.join(config.paths.save_name, subfolder_name),
            #     filename="predicted",
            #     n_class=config.test.n_class
            # )

            # visualize_ground_truth_timeline(
            #     df=df,
            #     save_dir=os.path.join(config.paths.save_name, subfolder_name),
            #     filename="ground_truth"
            # )

            # print(f"Visualization for folder {subfolder_name} completed.")
            
    
            
    
def test_treatment_classification_model(config):
    # setup
    device, num_gpus = setup_device()
    setup_seed(42)
    
    model = TreatmentClassificationModel(n_class=config.test.n_class,
                                         n_image=config.test.n_image,
                                         hidden_size=config.test.hidden_size,
                                         n_lstm=config.test.n_lstm
                                         )
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
            
    # CSVファイルの読み込み
    df = pd.read_csv(f'{config.paths.anomaly_test_csv}')

    # データの確認
    # print(df.head())
    
    # print(config.test.anomaly_class)
    # os._exit(0)
    

    # 各列の値の取り出し
    anomaly_test_results = {
        'image_paths': df['Image Path'].tolist(),
        'ground_truths': df['Ground Truths'].tolist(),
        'predicted_labels': df['Predicted Label'].tolist(),
        'top1_labels': df['Top1_Label'].tolist(),
        'top2_labels': df['Top2_Label'].tolist(),
        'top3_labels': df['Top3_Label'].tolist(),
        'top1_probs': df['Top1_Prob'].tolist(),
        'top2_probs': df['Top2_Prob'].tolist(),
        'top3_probs': df['Top3_Prob'].tolist()
    }
        
    # predicted_labels が4未満のサンプルを抽出
    if config.test.anomaly_class > 4:
        test_df = df[df['Predicted Label'] < 4]
        
    # predicted_labels が0のサンプルを抽出
    elif config.test.anomaly_class == 2:
        test_df = df[df['Predicted Label'] < 1]

    # 画像パスとラベルを取り出す
    test_image_paths = test_df['Image Path'].tolist()
    test_ground_truths = test_df['Ground Truths'].tolist()
    
    # print(test_image_paths)
    # print(len(test_image_paths))
    # print(len(test_ground_truths))
    # os._exit(0)
        
    # テストデータセットの作成
    test_dataset = [(image_path, ground_truth) for image_path, ground_truth in zip(test_image_paths, test_ground_truths)]
    
    # print(test_dataset[0])
    # os._exit(0)
    
    maker = 'olympus'
    test_folder_name = "20230803-110626-ES06_20230803-111315-es06-hd"
    
    test_datasets = TreatmentClassificationDatasetForTest(os.path.join(config.paths.root, maker, test_folder_name),
                                                          test_dataset,
                                                          config.test.n_image,
                                                          DataTransformer.get_transform(maker, 'test', config.test.img_size))
    
    # 可視化と保存
    # visualize_dataset(test_datasets, output_dir="visualized_samples")
    
    test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=4 * num_gpus)
        
    print(len(test_datasets))
    # os._exit(0)
    
    # 学習の経過を保存
    model.eval()
    
    all_probabilities = []
    all_predictions = []
    all_ground_truths = []
    all_image_paths = []
    
    # 各テストフォルダ毎の結果の保存folderを作成
    if not os.path.exists(os.path.join(config.paths.save_name, test_folder_name)):
        os.mkdir(os.path.join(config.paths.save_name, test_folder_name))
        
    evaluator = ModelEvaluator(os.path.join(config.paths.save_name, test_folder_name))
    
    # os._exit(0)
    
    with torch.no_grad():
        for images, image_path, label in test_dataloader:
            images = images.to(device)
            label = label.to(device)

            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = torch.max(probabilities, 1)
            
            all_probabilities.extend(probabilities.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_ground_truths.extend(label.cpu().numpy())
            all_image_paths.extend(image_path)
            
    evaluator.save_treatment_test_results(all_image_paths, all_ground_truths, all_predictions, all_probabilities, config.paths.anomaly_test_csv)
    
    
    
def parse_args():
    parser = argparse.ArgumentParser(description='Multi-class organ classification model using ResNet')
    parser.add_argument('-c', '--config', help='Path to config file', default='config.yaml')
    return parser.parse_args()
    
def main():
    args = parse_args()
    config = load_config(args.config)
        
    # 結果保存folderを作成
    if not os.path.exists(os.path.join(config.paths.save_name)):
        os.mkdir(os.path.join(config.paths.save_name))
    
    if config.test.mode =='anomaly_detection':
        test_anomaly_detection_model(config)
    elif config.test.mode =='treatment_classification':
        test_treatment_classification_model(config)
    else:
        raise ValueError("Invalid mode.")
    
    # Use the config in your code
    # test(config)

if __name__ == '__main__':
    main()