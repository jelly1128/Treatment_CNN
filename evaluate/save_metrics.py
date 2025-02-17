import csv
from pathlib import Path
import numpy as np

def save_metrics_to_csv(video_metrics, overall_metrics, video_file_path, overall_file_path):
    """
    メトリクスをCSVファイルに保存する関数

    Args:
        video_metrics (dict): 各動画のメトリクス
        overall_metrics (dict): 全体のメトリクス
        video_file_path (str): 各動画のメトリクスを保存するCSVファイルのパス
        overall_file_path (str): 全体のメトリクスを保存するCSVファイルのパス
    """
    # 各動画のメトリクスをCSVに保存
    with open(video_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Video", "Class", "Precision", "Recall", "Accuracy"])
        
        for video, metrics in video_metrics.items():
            for class_idx, (precision, recall, accuracy) in enumerate(zip(metrics['precision'], metrics['recall'], metrics['accuracy'])):
                writer.writerow([video, class_idx, precision, recall, accuracy])
    
    # 全体のメトリクスをCSVに保存
    with open(overall_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Class", "Precision", "Recall", "Accuracy"])
        
        for class_idx, (precision, recall, accuracy) in enumerate(zip(overall_metrics['precision'], overall_metrics['recall'], overall_metrics['accuracy'])):
            writer.writerow([class_idx, precision, recall, accuracy])
        
        writer.writerow(["Overall", overall_metrics['overall_precision'], overall_metrics['overall_recall'], overall_metrics['overall_accuracy']])

def save_video_multilabel_metrics_to_csv(video_metrics: dict[str, dict[str, float]], base_save_dir: Path, methods: str):
    """
    各動画フォルダにメトリクスをCSVファイルに保存する関数

    Args:
        video_metrics (dict): 各動画のメトリクス
        base_save_dir (Path): 保存するベースディレクトリのパス
    """
    for video, metrics in video_metrics.items():
        video_dir = base_save_dir / video
        video_dir.mkdir(parents=True, exist_ok=True)
        methods_dir = video_dir / methods
        methods_dir.mkdir(parents=True, exist_ok=True)

        # 混同行列を保存
        confusion_matrix_file = video_dir / methods / f'{methods}_confusion_matrix.csv'
        with open(confusion_matrix_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Class", "TP", "FP", "FN", "TN"])
            for class_idx, cm in enumerate(metrics['confusion_matrix']):
                tp, fp, fn, tn = cm.ravel()
                writer.writerow([class_idx, tp, fp, fn, tn])

        # メトリクスを保存
        metrics_file = video_dir / methods / f'{methods}_metrics.csv'
        with open(metrics_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Class", "Precision", "Recall", "Accuracy"])
            for class_idx, (precision, recall, accuracy) in enumerate(zip(metrics['precision'], metrics['recall'], metrics['accuracy'])):
                writer.writerow([class_idx, precision, recall, accuracy])

def save_overall_metrics_to_csv(overall_metrics, overall_file_path):
    """
    全体のメトリクスをCSVファイルに保存する関数

    Args:
        overall_metrics (dict): 全体のメトリクス
        overall_file_path (str): 全体のメトリクスを保存するCSVファイルのパス
    """
    # 混同行列を保存
    confusion_matrix_file = Path(overall_file_path).with_name('overall_confusion_matrix.csv')
    with open(confusion_matrix_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Class", "TP", "FP", "FN", "TN"])
        for class_idx, cm in enumerate(overall_metrics['confusion_matrix']):
            tp, fp, fn, tn = cm.ravel()
            writer.writerow([class_idx, tp, fp, fn, tn])

    # メトリクスを保存
    metrics_file = Path(overall_file_path).with_name('overall_metrics.csv')
    with open(metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Class", "Precision", "Recall", "Accuracy"])
        for class_idx, (precision, recall, accuracy) in enumerate(zip(overall_metrics['precision'], overall_metrics['recall'], overall_metrics['accuracy'])):
            writer.writerow([class_idx, precision, recall, accuracy])
        
        writer.writerow(["Overall", overall_metrics['overall_precision'], overall_metrics['overall_recall'], overall_metrics['overall_accuracy']])

def save_video_metrics_to_csv_single(video_metrics, base_save_dir):
    """
    各動画フォルダにシングルラベルのメトリクスをCSVファイルに保存する関数

    Args:
        video_metrics (dict): 各動画のメトリクス
        base_save_dir (Path): 保存するベースディレクトリのパス
    """
    for video, metrics in video_metrics.items():
        video_dir = base_save_dir / video
        video_dir.mkdir(parents=True, exist_ok=True)

        # 混同行列を保存
        confusion_matrix_file = video_dir / 'confusion_matrix_single.csv'
        with open(confusion_matrix_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([""] + [f"Pred_{i}" for i in range(metrics['confusion_matrix'].shape[0])])
            for i, row in enumerate(metrics['confusion_matrix']):
                writer.writerow([f"True_{i}"] + row.tolist())

        # メトリクスを保存
        metrics_file = video_dir / 'metrics_single.csv'
        with open(metrics_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Class", "Precision", "Recall", "Accuracy"])
            for class_idx, (precision, recall, accuracy) in enumerate(zip(metrics['precision'], metrics['recall'], metrics['accuracy'])):
                writer.writerow([class_idx, precision, recall, accuracy])

def save_overall_metrics_to_csv_single(overall_metrics, overall_file_path):
    """
    全体のシングルラベルのメトリクスをCSVファイルに保存する関数

    Args:
        overall_metrics (dict): 全体のメトリクス
        overall_file_path (str): 全体のメトリクスを保存するCSVファイルのパス
    """
    # 混同行列を保存
    confusion_matrix_file = Path(overall_file_path).with_name('overall_confusion_matrix_single.csv')
    with open(confusion_matrix_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([""] + [f"Pred_{i}" for i in range(overall_metrics['confusion_matrix'].shape[0])])
        for i, row in enumerate(overall_metrics['confusion_matrix']):
            writer.writerow([f"True_{i}"] + row.tolist())

    # メトリクスを保存
    metrics_file = Path(overall_file_path).with_name('overall_metrics_single.csv')
    with open(metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Class", "Precision", "Recall", "Accuracy"])
        for class_idx, (precision, recall, accuracy) in enumerate(zip(overall_metrics['precision'], overall_metrics['recall'], overall_metrics['accuracy'])):
            writer.writerow([class_idx, precision, recall, accuracy]) 