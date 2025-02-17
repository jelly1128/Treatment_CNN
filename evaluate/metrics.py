from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix
import numpy as np
import csv
from labeling.label_converter import HardMultiLabelResult, SingleLabelResult
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
        """
        各クラスごとのTP, FP, TN, FNを計算
        
        Args:
            y_true: one-hot形式の正解ラベル (n_samples, n_classes)
            y_pred: one-hot形式の予測ラベル (n_samples, n_classes)
            
        Returns:
            precision: 各クラスの適合率
            recall: 各クラスの再現率
            accuracy: 各クラスの正解率
            confusion_matrices: 各クラスの混同行列
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        n_classes = y_true.shape[1]
        
        precisions = []
        recalls = []
        accuracies = []
        confusion_matrices = []
        
        for class_idx in range(n_classes):
            tp = np.sum((y_true[:, class_idx] == 1) & (y_pred[:, class_idx] == 1))
            fp = np.sum((y_true[:, class_idx] == 0) & (y_pred[:, class_idx] == 1))
            tn = np.sum((y_true[:, class_idx] == 0) & (y_pred[:, class_idx] == 0))
            fn = np.sum((y_true[:, class_idx] == 1) & (y_pred[:, class_idx] == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            accuracies.append(accuracy)
            confusion_matrices.append([[tp, fp], [fn, tn]])
        
        return np.array(precisions), np.array(recalls), np.array(accuracies), np.array(confusion_matrices)
        
        # # y_trueとy_predが2次元配列であることを確認
        # y_true = np.atleast_2d(y_true)
        # y_pred = np.atleast_2d(y_pred)

        # precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        # recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        # accuracy = np.mean(y_true == y_pred, axis=0)  # 各クラスごとの正解率
        # cm = self.calculate_confusion_matrix(y_true, y_pred)
        # return precision, recall, accuracy, cm

    def calculate_multilabel_metrics_per_video(self, hard_multilabel_results: dict[str, HardMultiLabelResult]) -> dict[str, dict[str, float]]:
        """
        各動画単位の混同行列と適合率・再現率・正解率を計算する関数

        Args:
            hard_multilabel_results (dict[str, HardMultiLabelResult]): 各フォルダのマルチラベルの結果

        Returns:
            dict: 各動画のメトリクスを格納した辞書
        """
        video_metrics = {}

        for folder_name, hard_multilabel_result in hard_multilabel_results.items():
            y_true = np.array(hard_multilabel_result.ground_truth_labels)
            y_pred = np.array(hard_multilabel_result.multilabels)
            
            precision, recall, accuracy, cm = self.calculate_metrics_per_class(y_true, y_pred)
            video_metrics[folder_name] = {
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'confusion_matrix': cm
            }
        
        return video_metrics
        # """
        # 各動画単位の混同行列と適合率・再現率・正解率を計算する関数

        # Args:
        #     hard_multilabel_results (dict[str, HardMultiLabelResult]): 各フォルダのマルチラベルの結果

        # Returns:
        #     dict: 各動画のメトリクスを格納した辞書
        # """
        # video_metrics = {}

        # for folder_name, hard_multilabel_result in hard_multilabel_results.items():
        #     y_true = np.array(hard_multilabel_result.ground_truth_labels)
        #     y_pred = np.array(hard_multilabel_result.multilabels)
            
        #     precision, recall, accuracy, cm = self.calculate_metrics_per_class(y_true, y_pred)
        #     video_metrics[folder_name] = {
        #         'precision': precision,
        #         'recall': recall,
        #         'accuracy': accuracy,
        #         'confusion_matrix': cm
        #     }
        
        # return video_metrics

    def calculate_multilabel_overall_metrics(self, hard_multilabel_results: dict[str, HardMultiLabelResult]):
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

        # クラスごとの混同行列を計算
        confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)
        
        # 全体の混同行列を計算（全クラスの合計）
        total_tp = np.sum([cm[1, 1] for cm in confusion_matrices])
        total_fp = np.sum([cm[0, 1] for cm in confusion_matrices])
        total_tn = np.sum([cm[0, 0] for cm in confusion_matrices])
        total_fn = np.sum([cm[1, 0] for cm in confusion_matrices])
        
        # 全体の精度指標を計算
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn) if (total_tp + total_fp + total_tn + total_fn) > 0 else 0
    
        return {
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_accuracy': overall_accuracy,
            'total_TP': total_tp,
            'total_FP': total_fp,
            'total_TN': total_tn,
            'total_FN': total_fn
        }
        # """
        # 全動画の混同行列と適合率・再現率・正解率を計算する関数

        # Args:
        #     hard_multilabel_results (dict[str, HardMultiLabelResult]): 各フォルダのマルチラベルの結果

        # Returns:
        #     dict: 全動画のメトリクスを格納した辞書
        # """
        # y_true = []
        # y_pred = []

        # for hard_multilabel_result in hard_multilabel_results.values():
        #     y_true.extend(hard_multilabel_result.ground_truth_labels)
        #     y_pred.extend(hard_multilabel_result.multilabels)
        
        # y_true = np.array(y_true)
        # y_pred = np.array(y_pred)

        # precision, recall, accuracy, cm = self.calculate_metrics_per_class(y_true, y_pred)
        # overall_metrics = self.calculate(y_true, y_pred)
        
        # return {
        #     'precision': precision,
        #     'recall': recall,
        #     'accuracy': accuracy,
        #     'overall_precision': overall_metrics['precision'],
        #     'overall_recall': overall_metrics['recall'],
        #     'overall_accuracy': overall_metrics['accuracy'],
        #     'confusion_matrix': cm
        # }