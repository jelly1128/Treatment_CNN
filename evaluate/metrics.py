from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix
import numpy as np
import csv
from labeling.label_converter import HardMultiLabelResult
class ClassificationMetricsCalculator:
    def calculate(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
    
    def calculate_confusion_matrix(self, y_true, y_pred):
        return multilabel_confusion_matrix(y_true, y_pred)
    
    def calculate_metrics_per_class(self, y_true, y_pred):
        # y_trueとy_predが2次元配列であることを確認
        y_true = np.atleast_2d(y_true)
        y_pred = np.atleast_2d(y_pred)

        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        accuracy = np.mean(y_true == y_pred, axis=0)  # 各クラスごとの正解率
        cm = self.calculate_confusion_matrix(y_true, y_pred)
        return precision, recall, accuracy, cm

    def calculate_metrics_per_video(self, hard_multilabel_results: dict[str, HardMultiLabelResult]):
        """
        各動画単位の混同行列と適合率・再現率・正解率を計算する関数

        Args:
            hard_multilabel_results (dict[str, HardMultiLabelResult]): 各フォルダのマルチラベルの結果

        Returns:
            dict: 各動画のメトリクスを格納した辞書
        """
        video_metrics = {}

        for folder_name, hard_multilabel_result in hard_multilabel_results.items():
            for idx, (y_true, y_pred) in enumerate(zip(hard_multilabel_result.ground_truth_labels, hard_multilabel_result.multilabels)):
                precision, recall, accuracy, cm = self.calculate_metrics_per_class([y_true], [y_pred])
                video_metrics[f"{folder_name}_{idx}"] = {
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy,
                    'confusion_matrix': cm
                }
        
        return video_metrics

    def calculate_overall_metrics(self, hard_multilabel_results: dict[str, HardMultiLabelResult]):
        """
        全動画の混同行列と適合率・再現率・正解率を計算する関数

        Args:
            hard_multilabel_results (dict[str, HardMultiLabelResult]): 各フォルダのマルチラベルの結果

        Returns:
            dict: 全動画のメトリクスを格納した辞書
        """
        y_true = []
        y_pred = []

        for hard_multilabel_result in hard_multilabel_results.values():
            y_true.extend(hard_multilabel_result.ground_truth_labels)
            y_pred.extend(hard_multilabel_result.multilabels)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        precision, recall, accuracy, cm = self.calculate_metrics_per_class(y_true, y_pred)
        overall_metrics = self.calculate(y_true, y_pred)
        
        return {
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'overall_precision': overall_metrics['precision'],
            'overall_recall': overall_metrics['recall'],
            'overall_accuracy': overall_metrics['accuracy'],
            'confusion_matrix': cm
        }