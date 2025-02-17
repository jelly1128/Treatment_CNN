import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import logging
import csv
from PIL import Image, ImageDraw
from engine.inference import InferenceResult
from labeling.label_converter import HardMultiLabelResult, SingleLabelResult

# 定数の定義
LABEL_COLORS = {
    0: (254, 195, 195),  # white
    1: (204, 66, 38),    # lugol 
    2: (57, 103, 177),   # indigo
    3: (96, 165, 53),    # nbi
    4: (86, 65, 72),     # outside
    5: (159, 190, 183),  # bucket
}
DEFAULT_COLOR = (148, 148, 148)

class ResultsVisualizer:
    def __init__(self, save_dir: Path):
        """
        結果の可視化を行うクラス

        Args:
            save_dir: 可視化結果の保存ディレクトリ
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def load_results(self, csv_path: Path) -> InferenceResult:
        """
        CSVファイルから推論結果を読み込む。

        Args:
            csv_path: 読み込むCSVファイルのパス

        Returns:
            InferenceResult: 読み込んだ推論結果
        """
        try:
            image_paths, probabilities, labels = self._read_csv(csv_path)
            return InferenceResult(image_paths=image_paths, probabilities=probabilities, labels=labels)
        except Exception as e:
            logging.error(f"Failed to load results from {csv_path}: {e}")
            raise
    
    def _read_csv(self, csv_path: Path):
        """CSVファイルを読み込み、データを分割するヘルパーメソッド"""
        image_paths = []
        probabilities = []
        labels = []

        with open(csv_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)  # ヘッダーをスキップ

            for row in reader:
                image_paths.append(row[0])
                num_probs = (len(row) - 1) // 2
                probs = list(map(float, row[1:num_probs + 1]))
                lbls = list(map(int, row[num_probs + 1:]))
                probabilities.append(probs)
                labels.append(lbls)

        logging.info(f"Loaded results from {csv_path}")
        return image_paths, probabilities, labels
      
    def save_multilabel_visualization(self, results: dict[str, HardMultiLabelResult], save_path: Path = None, methods: str = 'multilabel'):
        """マルチラベル分類の予測結果を時系列で可視化"""
        
        for folder_name, result in results.items():
            # マルチラベルの予測結果を取得
            predicted_labels = np.array(result.multilabels)
            n_images = len(predicted_labels)
            n_classes = len(predicted_labels[0])
            
            # 時系列の画像を作成
            timeline_width = n_images
            timeline_height = n_classes * (n_images // 10)
            
            timeline_image = Image.new('RGB', (timeline_width, timeline_height), (255, 255, 255))
            draw = ImageDraw.Draw(timeline_image)
            
            for i in range(n_images):
                labels = predicted_labels[i]
                for label_idx, label_value in enumerate(labels):
                    if label_value == 1:
                        row_idx = label_idx
                        
                        x1 = i * (timeline_width // n_images)
                        x2 = (i + 1) * (timeline_width // n_images)
                        y1 = row_idx * (n_images // 10)
                        y2 = (row_idx + 1) * (n_images // 10)

                        color = LABEL_COLORS.get(label_idx, DEFAULT_COLOR)
                        draw.rectangle([x1, y1, x2, y2], fill=color)

            # 保存パスを正しく設定
            if save_path is None:
                save_dir = self.save_dir / folder_name / methods
                save_dir.mkdir(parents=True, exist_ok=True)
                save_file = save_dir / f'multilabel_{folder_name}.png'
            else:
                save_path = save_path / methods
                save_path.mkdir(parents=True, exist_ok=True)  # ディレクトリが存在しない場合は作成
                save_file = save_path / f'multilabel_{folder_name}.png'
            
            timeline_image.save(save_file)
            logging.info(f'Timeline image saved at {save_file}')
            
    def save_singlelabel_visualization(self, results: dict[str, SingleLabelResult], save_path: Path = None):
        """シングルラベル分類の予測結果を時系列で可視化"""
        for folder_name, result in results.items():
            # シングルラベルの予測結果を取得
            predicted_labels = np.array(result.single_labels)
            n_images = len(predicted_labels)
            
            # 時系列の画像を作成
            timeline_width = n_images
            timeline_height = n_images // 10
            
            timeline_image = Image.new('RGB', (timeline_width, timeline_height), (255, 255, 255))
            draw = ImageDraw.Draw(timeline_image)
            
            for i in range(n_images):
                label = predicted_labels[i]
                x1 = i * (timeline_width // n_images)
                x2 = (i + 1) * (timeline_width // n_images)
                y1 = label * (n_images // 10)
                y2 = (label + 1) * (n_images // 10)
                
                color = LABEL_COLORS.get(label, DEFAULT_COLOR)
                draw.rectangle([x1, y1, x2, y2], fill=color)
                
            # 保存パスを正しく設定
            if save_path is None:
                save_dir = self.save_dir / folder_name
                save_dir.mkdir(parents=True, exist_ok=True)
                save_file = save_dir / f'{folder_name}.png'
            else:
                save_path.mkdir(parents=True, exist_ok=True)  # ディレクトリが存在しない場合は作成
                save_file = save_path / f'singlelabel_{folder_name}.png'
                
            timeline_image.save(save_file)
            logging.info(f'Timeline image saved at {save_file}')
            
    def save_main_classes_visualization(self, results: dict[str, HardMultiLabelResult], save_path: Path = None):
        """正解ラベルを時系列で可視化"""
        for folder_name, result in results.items():
            # 主クラス（0-5）のみを抽出
            ground_truth_labels = np.array(result.ground_truth_labels)[:, :6]  # 最初の6クラスのみを抽出
            n_images = len(ground_truth_labels)
            n_classes = 6  # 主クラスの数
            
            # 時系列の画像を作成
            timeline_width = n_images
            timeline_height = n_images // 10
            
            timeline_image = Image.new('RGB', (timeline_width, timeline_height), (255, 255, 255))
            draw = ImageDraw.Draw(timeline_image)
            
            for i in range(n_images):
                labels = ground_truth_labels[i]
                for label_idx, label_value in enumerate(labels):
                    if label_value == 1:
                        x1 = i * (timeline_width // n_images)
                        x2 = (i + 1) * (timeline_width // n_images)

                        color = LABEL_COLORS.get(label_idx, DEFAULT_COLOR)
                        draw.rectangle([x1, 0, x2, timeline_height], fill=color)

            # 保存パスを正しく設定
            if save_path is None:
                save_dir = self.save_dir / folder_name
                save_dir.mkdir(parents=True, exist_ok=True)
                save_file = save_dir / f'main_classes_{folder_name}.png'
            else:
                save_path.mkdir(parents=True, exist_ok=True)
                save_file = save_path / f'main_classes_{folder_name}.png'
            
            timeline_image.save(save_file)
            logging.info(f'Main classes timeline image saved at {save_file}')
            
    # def plot_confusion_matrices(self, predictions: np.ndarray, labels: np.ndarray, 
    #                           class_names: list = None):
    #     """各クラスの混同行列をプロット"""
    #     num_classes = labels.shape[1]
    #     class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        
    #     for i in range(num_classes):
    #         # 予測確率を二値化
    #         binary_preds = (predictions[:, i] > 0.5).astype(int)
    #         binary_labels = labels[:, i]
            
    #         cm = confusion_matrix(binary_labels, binary_preds)
            
    #         plt.figure(figsize=(8, 6))
    #         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    #         plt.title(f'Confusion Matrix - {class_names[i]}')
    #         plt.ylabel('True Label')
    #         plt.xlabel('Predicted Label')
    #         plt.savefig(self.save_dir / f'confusion_matrix_class_{i}.png')
    #         plt.close()
            
    #     logging.info(f"Saved confusion matrices to {self.save_dir}")

    # def plot_precision_recall_curves(self, predictions: np.ndarray, labels: np.ndarray,
    #                                class_names: list = None):
    #     """各クラスのPrecision-Recallカーブをプロット"""
    #     num_classes = labels.shape[1]
    #     class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        
    #     plt.figure(figsize=(10, 8))
        
    #     for i in range(num_classes):
    #         precision, recall, _ = precision_recall_curve(labels[:, i], predictions[:, i])
    #         ap = average_precision_score(labels[:, i], predictions[:, i])
            
    #         plt.plot(recall, precision, label=f'{class_names[i]} (AP={ap:.2f})')
        
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.title('Precision-Recall Curves')
    #     plt.legend(loc='lower left')
    #     plt.grid(True)
    #     plt.savefig(self.save_dir / 'precision_recall_curves.png')
    #     plt.close()
        
    #     logging.info(f"Saved precision-recall curves to {self.save_dir}")

    # def plot_prediction_distribution(self, predictions: np.ndarray, class_names: list = None):
    #     """予測確率の分布をプロット"""
    #     num_classes = predictions.shape[1]
    #     class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        
    #     plt.figure(figsize=(12, 6))
        
    #     for i in range(num_classes):
    #         sns.kdeplot(predictions[:, i], label=class_names[i])
        
    #     plt.xlabel('Prediction Probability')
    #     plt.ylabel('Density')
    #     plt.title('Distribution of Prediction Probabilities')
    #     plt.legend()
    #     plt.savefig(self.save_dir / 'prediction_distribution.png')
    #     plt.close()
        
    #     logging.info(f"Saved prediction distribution plot to {self.save_dir}")

    # def plot_correct_incorrect_predictions(self, predictions: np.ndarray, labels: np.ndarray,
    #                                      class_names: list = None):
    #     """正しい予測と誤った予測の分布を比較"""
    #     num_classes = labels.shape[1]
    #     class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        
    #     for i in range(num_classes):
    #         plt.figure(figsize=(10, 6))
    #         binary_preds = (predictions[:, i] > 0.5).astype(int)
            
    #         correct_preds = predictions[binary_preds == labels[:, i], i]
    #         incorrect_preds = predictions[binary_preds != labels[:, i], i]
            
    #         sns.kdeplot(correct_preds, label='Correct Predictions')
    #         sns.kdeplot(incorrect_preds, label='Incorrect Predictions')
            
    #         plt.title(f'Prediction Distribution - {class_names[i]}')
    #         plt.xlabel('Prediction Probability')
    #         plt.ylabel('Density')
    #         plt.legend()
    #         plt.savefig(self.save_dir / f'pred_distribution_class_{i}.png')
    #         plt.close()
        
    #     logging.info(f"Saved correct/incorrect prediction distributions to {self.save_dir}")

    # def plot_metrics_summary(self, metrics_dict: dict):
    #     """評価指標のサマリーを可視化"""
    #     plt.figure(figsize=(12, 6))
        
    #     metrics = list(metrics_dict.values())
    #     labels = list(metrics_dict.keys())
        
    #     sns.barplot(x=labels, y=metrics)
    #     plt.xticks(rotation=45)
    #     plt.title('Performance Metrics Summary')
    #     plt.tight_layout()
    #     plt.savefig(self.save_dir / 'metrics_summary.png')
    #     plt.close()
        
    #     logging.info(f"Saved metrics summary to {self.save_dir}")

    # def visualize_all(self, predictions: np.ndarray, labels: np.ndarray,
    #                  metrics_dict: dict = None, class_names: list = None):
    #     """全ての可視化を一括実行"""
    #     self.plot_confusion_matrices(predictions, labels, class_names)
    #     self.plot_precision_recall_curves(predictions, labels, class_names)
    #     self.plot_prediction_distribution(predictions, class_names)
    #     self.plot_correct_incorrect_predictions(predictions, labels, class_names)
        
    #     if metrics_dict is not None:
    #         self.plot_metrics_summary(metrics_dict)
        
    #     logging.info("Generated all visualization plots")