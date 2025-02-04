import os
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging
from PIL import Image, ImageDraw

class Analyzer:
    def __init__(self, save_dir, num_classes):
        """
        推論結果を解析するクラス。

        Args:
            save_dir (str): 結果を保存するディレクトリ
            num_classes (int): クラス数
        """
        self.save_dir = save_dir
        self.num_classes = num_classes

    def analyze(self, results):
        """
        推論結果を解析し、CSVファイルに保存。

        Args:
            results (dict): 推論結果 (Inference クラスの出力)
        """
        for folder_name, (probabilities, labels, image_paths) in results.items():
            logging.info(f"Analyzing results for {folder_name}...")

            folder_path = os.path.join(self.save_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            self.save_raw_results(folder_path, image_paths, probabilities, labels)
            pred_labels = self.threshold_predictions(probabilities)
            self.save_threshold_results(folder_path, image_paths, pred_labels, labels)
            self.calculate_metrics(folder_path, pred_labels, labels)
            
            # DataFrameの生成
            columns_predicted = [f"Predicted_Class_{i}" for i in range(self.num_classes)]
            columns_labels = [f"Label_Class_{i}" for i in range(self.num_classes)]
            df = pd.DataFrame(
                data=np.hstack([pred_labels, np.array(labels)]),
                columns=columns_predicted + columns_labels
            )
            df['Image_Path'] = image_paths  # 画像パスを追加
            
            # print("df.columns:", df.columns)  # データフレームのカラム一覧を出力
            # print(df.head())  # 最初の数行を出力

            self.visualize_multilabel_timeline(df, folder_path, folder_name)
            self.visualize_ground_truth_timeline(df, folder_path, folder_name)
            
    def save_raw_results(self, folder_path, image_paths, probabilities, labels):
        """生の推論結果（確率値）とラベルをCSVに保存。"""
        csv_path = os.path.join(folder_path, 'raw_results.csv')
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Image_Path'] + [f"Pred_Class_{i}" for i in range(self.num_classes)] + [f"True_Class_{i}" for i in range(self.num_classes)]
            writer.writerow(header)
            for img_path, probs, lbls in zip(image_paths, probabilities, labels):
                writer.writerow([img_path] + probs + lbls.tolist())
        logging.info(f"Saved raw results: {csv_path}")

    def threshold_predictions(self, probabilities):
        """
        50%閾値で予測ラベルを作成。

        50%以上の確率を持つクラスを 1, 50%以下の確率を持つクラスを 0 として予測ラベルを作成。
        ただし、どのクラスも 50%以下の確率を持つ場合、最も確率が高いクラスを 1 に設定。
        """
        pred_labels = (np.array(probabilities) >= 0.5).astype(int)

        # どのクラスも 50%以下の確率を持つ場合、最も確率が高いクラスを 1 に設定
        for i, probs in enumerate(probabilities):
            if all(prob < 0.5 for prob in probs):
                max_prob_index = np.argmax(probs)
                pred_labels[i][max_prob_index] = 1  # 最も確率が高いクラスを 1 に設定

        return pred_labels

    def save_threshold_results(self, folder_path, image_paths, pred_labels, labels):
        """閾値を適用した予測ラベルを保存。"""
        csv_path = os.path.join(folder_path, 'threshold_results.csv')
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Image_Path'] + [f"Pred_Class_{i}" for i in range(self.num_classes)] + [f"True_Class_{i}" for i in range(self.num_classes)]
            writer.writerow(header)
            for img_path, pred, true in zip(image_paths, pred_labels, labels):
                writer.writerow([img_path] + pred.tolist() + true.tolist())
        logging.info(f"Saved threshold results: {csv_path}")

    def calculate_metrics(self, folder_path, pred_labels, labels):
        """評価指標（精度・適合率・再現率・F1スコア）を計算しCSVに保存。"""
        metrics_csv_path = os.path.join(folder_path, 'metrics.csv')
        conf_matrix_csv_path = os.path.join(folder_path, 'confusion_matrix.csv')

        with open(metrics_csv_path, mode='w', newline='') as metrics_file, open(conf_matrix_csv_path, mode='w', newline='') as conf_matrix_file:
            metrics_writer = csv.writer(metrics_file)
            conf_matrix_writer = csv.writer(conf_matrix_file)

            metrics_writer.writerow(['Class', 'Accuracy', 'Precision', 'Recall', 'F1_Score'])
            conf_matrix_writer.writerow(['Class', 'TP', 'FP', 'TN', 'FN'])

            for class_idx in range(self.num_classes):
                true_labels = [lbl[class_idx] for lbl in labels]
                pred_class_labels = pred_labels[:, class_idx]

                accuracy = accuracy_score(true_labels, pred_class_labels)
                precision = precision_score(true_labels, pred_class_labels, zero_division=0)
                recall = recall_score(true_labels, pred_class_labels, zero_division=0)
                f1 = f1_score(true_labels, pred_class_labels, zero_division=0)

                metrics_writer.writerow([class_idx, f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])

                conf_matrix = confusion_matrix(true_labels, pred_class_labels, labels=[0, 1])
                tn, fp, fn, tp = conf_matrix.ravel()

                conf_matrix_writer.writerow([class_idx, tp, fp, tn, fn])

                logging.info(f"Class {class_idx} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

        logging.info(f"Saved metrics: {metrics_csv_path}")
        logging.info(f"Saved confusion matrix: {conf_matrix_csv_path}")

    def visualize_multilabel_timeline(self, df, save_dir, filename):
        """マルチラベル分類の予測結果を時系列で可視化"""
        label_colors = {
            0: (254, 195, 195),  # white
            1: (204, 66, 38),    # lugol
            2: (57, 103, 177),   # indigo
            3: (96, 165, 53),    # nbi
            4: (86, 65, 72),     # outside
            5: (159, 190, 183),  # bucket
        }
        default_color = (148, 148, 148)

        predicted_labels = df[[f"Predicted_Class_{i}" for i in range(self.num_classes)]].values

        n_images = len(predicted_labels)

        timeline_width = n_images
        timeline_height = self.num_classes * (n_images // 10)

        timeline_image = Image.new('RGB', (timeline_width, timeline_height), (255, 255, 255))
        draw = ImageDraw.Draw(timeline_image)

        for i in range(n_images):
            labels = predicted_labels[i]
            for label_idx, label_value in enumerate(labels):
                if label_value == 1:
                    row_idx = label_idx
                    
                    # Calculate the position in the timeline
                    x1 = i * (timeline_width // n_images)
                    x2 = (i + 1) * (timeline_width // n_images)
                    y1 = row_idx * (n_images // 10)
                    y2 = (row_idx + 1) * (n_images // 10)

                    color = label_colors.get(label_idx, default_color)
                    draw.rectangle([x1, y1, x2, y2], fill=color)

        timeline_image.save(os.path.join(save_dir, f"{filename}_predicted_timeline.png"))
        logging.info(f'Timeline image saved at {os.path.join(save_dir, "multilabel_timeline.png")}')


    def visualize_ground_truth_timeline(self, df, save_dir, filename):
        """正解ラベルを時系列で可視化"""

        label_colors = {
            0: (254, 195, 195),
            1: (204, 66, 38),
            2: (57, 103, 177),
            3: (96, 165, 53),
            4: (86, 65, 72),
            5: (159, 190, 183),
        }
        default_color = (148, 148, 148)

        ground_truth_labels = df[[f"Label_Class_{i}" for i in range(self.num_classes)]].values
        n_images = len(ground_truth_labels)

        timeline_width = n_images
        timeline_height = 2 * (n_images // 10)  # 20 pixels per label row (2 rows total)

        timeline_image = Image.new('RGB', (timeline_width, timeline_height), (255, 255, 255))
        draw = ImageDraw.Draw(timeline_image)

        for i in range(n_images):
            labels = ground_truth_labels[i]
            for label_idx, label_value in enumerate(labels):
                if label_value == 1:
                    # Determine the correct row for drawing
                    row_idx = 0 if label_idx < 6 else 1

                    # Calculate the position in the timeline
                    x1 = i * (timeline_width // n_images)
                    x2 = (i + 1) * (timeline_width // n_images)
                    y1 = row_idx * (n_images // 10)  # Each row is 20 pixels tall
                    y2 = (row_idx + 1) * (n_images // 10)  # Height for the rectangle

                    color = label_colors.get(label_idx, default_color)
                    draw.rectangle([x1, y1, x2, y2], fill=color)

        timeline_image.save(os.path.join(save_dir, f"{filename}_ground_truth_timeline.png"))
        logging.info(f'Ground truth timeline image saved at {os.path.join(save_dir, f"{filename}_ground_truth_timeline.png")}')