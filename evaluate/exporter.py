import csv
from pathlib import Path
import numpy as np
import pandas as pd

def save_video_metrics_to_csv(video_metrics: dict[str, dict[str, float]], base_save_dir: Path, methods: str):
    """
    各動画フォルダにメトリクスをCSVファイルに保存する関数

    Args:
        video_metrics (dict): 各動画のメトリクス
        base_save_dir (Path): 保存するベースディレクトリのパス
    """
    for video_name, metrics in video_metrics.items():
        video_methods_results_dir = base_save_dir / video_name / methods
        video_methods_results_dir.mkdir(parents=True, exist_ok=True)

        # 混同行列を保存
        confusion_matrix_file = video_methods_results_dir / f'{methods}_confusion_matrix.csv'
        with open(confusion_matrix_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Class", "TP", "FP", "FN", "TN"])
            for class_idx, cm in enumerate(metrics['confusion_matrix']):
                tp, fp, fn, tn = cm.ravel()
                writer.writerow([class_idx, tp, fp, fn, tn])

        # メトリクスを保存
        metrics_file = video_methods_results_dir / f'{methods}_metrics.csv'
        with open(metrics_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Class", "Precision", "Recall", "F1 score", "Accuracy"])
            for class_idx, (precision, recall, f1_score, accuracy) in enumerate(zip(metrics['precision'], metrics['recall'], metrics['f1_score'], metrics['accuracy'])):
                writer.writerow([class_idx,
                                 f"{precision:.4f}",
                                 f"{recall:.4f}",
                                 f"{f1_score:.4f}",
                                 f"{accuracy:.4f}"]
                                 )

def save_overall_metrics_to_csv(overall_metrics, base_save_dir: Path, methods: str):
    """
    全体のメトリクスをCSVファイルに保存する関数

    Args:
        overall_metrics (dict): 全体のメトリクス
        overall_file_path (str): 保存先のベースパス
        methods (str): 評価手法の名前
    """
    base_path = base_save_dir / methods
    base_path.mkdir(parents=True, exist_ok=True)

    # 各クラスの精度指標を保存
    class_metrics_file = base_path / f'{methods}_class_metrics.csv'
    with open(class_metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'Precision', 'Recall', "F1 score", 'Accuracy'])
        for class_idx, metrics in enumerate(overall_metrics['class_metrics']):
            writer.writerow([class_idx,
                             f"{metrics['precision']:.4f}",
                             f"{metrics['recall']:.4f}",
                             f"{metrics['f1_score']:.4f}",
                             f"{metrics['accuracy']:.4f}"
                            ])

    # 各クラスの2×2混同行列を保存
    per_class_cm_file = base_path / f'{methods}_per_class_confusion_matrices.csv'
    with open(per_class_cm_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'TP', 'FP', 'FN', 'TN'])
        for class_idx, cm in enumerate(overall_metrics['per_class_confusion_matrices']):
            tp, fp = cm[1, 1], cm[0, 1]
            fn, tn = cm[1, 0], cm[0, 0]
            writer.writerow([class_idx, tp, fp, fn, tn])

    # クラス数×クラス数の混同行列を保存
    class_cm_file = base_path / f'{methods}_overall_class_confusion_matrix.csv'
    with open(class_cm_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # ヘッダー行
        writer.writerow(['True/Pred'] + [f'Pred_{i}' for i in range(len(overall_metrics['class_confusion_matrix']))])
        # データ行
        for i, row in enumerate(overall_metrics['class_confusion_matrix']):
            writer.writerow([f'True_{i}'] + row.tolist())

def save_hard_multi_label_results_to_csv(
    hard_multi_label_results: dict,
    save_dir_path: Path,
    methods: str
):
    """
    マルチラベルの変換結果をCSVファイルに保存する関数

    Args:
        hard_multi_label_results (dict): 変換結果の辞書
        save_dir_path (Path): 保存先ベースディレクトリ
        methods (str): 変換手法の名前（ファイル名に使用）
    """
    save_dir_path.mkdir(parents=True, exist_ok=True)

    for video_name, result in hard_multi_label_results.items():
        video_results_dir = save_dir_path / video_name / methods
        video_results_dir.mkdir(parents=True, exist_ok=True)
        video_results_csv = video_results_dir / f'{methods}_results_{video_name}.csv'

        with open(video_results_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Image_Path']
            header.extend([f'Label_{i}' for i in range(len(result.multi_labels[0]))])
            header.extend([f'True_Label_{i}' for i in range(len(result.ground_truth_labels[0]))])
            writer.writerow(header)

            for img_path, pred_labels, true_labels in zip(
                result.image_paths, result.multi_labels, result.ground_truth_labels
            ):
                row = [img_path]
                row.extend(pred_labels)
                row.extend(true_labels)
                writer.writerow(row)

def save_single_label_results_to_csv(
    single_label_results: dict,
    save_dir_path: Path,
    methods: str
):
    """
    シングルラベル予測結果をCSVファイルに保存する関数

    Args:
        single_label_results (dict): シングルラベル結果の辞書
        save_dir_path (Path): 保存先ベースディレクトリ
        methods (str): 手法の名前（ファイル名に使用）
    """
    for folder_name, result in single_label_results.items():
        save_path = save_dir_path / folder_name / methods
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / f'{methods}_results_{folder_name}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Image_Path', 'Pred_Class', 'True_Class'])
            for image_path, pred_label, true_label in zip(
                result.image_paths, result.single_labels, result.ground_truth_labels
            ):
                writer.writerow([image_path, pred_label, true_label])

def save_raw_inference_results_to_csv(result, save_dir_path: Path, video_name: str):
    """
    推論生結果をCSVファイルに保存する関数

    Args:
        result: InferenceResult（image_paths, probabilities, labels）
        save_dir_path (Path): 保存先ベースディレクトリ
        video_name (str): 動画名（サブディレクトリ名およびファイル名に使用）
    """
    video_dir = save_dir_path / video_name
    video_dir.mkdir(parents=True, exist_ok=True)

    csv_path = video_dir / f'raw_results_{video_name}.csv'
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Image_Path']
        header += [f'Pred_Class_{i}' for i in range(len(result.probabilities[0]))]
        header += [f'True_Class_{i}' for i in range(len(result.labels[0]))]
        writer.writerow(header)
        for img_path, probs, label in zip(result.image_paths, result.probabilities, result.labels):
            writer.writerow([img_path] + probs + label)


def save_all_folds_metrics_to_csv(
    class_metrics: list,
    class_confusion_matrix,
    save_dir: Path
):
    """
    全fold統合のメトリクスをCSVファイルに保存する関数

    Args:
        class_metrics (list): 各クラスのメトリクス辞書のリスト
        class_confusion_matrix: クラス数×クラス数の混同行列
        save_dir (Path): 保存先ディレクトリ
    """
    n_classes = len(class_metrics)

    cm_df = pd.DataFrame(
        class_confusion_matrix,
        index=[f'True_{i}' for i in range(n_classes)],
        columns=[f'Pred_{i}' for i in range(n_classes)]
    )
    cm_df.to_csv(save_dir / 'confusion_matrix.csv')

    metrics_df = pd.DataFrame([
        {
            'Class': i,
            'Precision': round(metrics['precision'], 4),
            'Recall': round(metrics['recall'], 4),
            'F1 Score': round(metrics['f1_score'], 4),
            'Accuracy': round(metrics['accuracy'], 4)
        }
        for i, metrics in enumerate(class_metrics)
    ])
    metrics_df.to_csv(save_dir / 'class_metrics.csv', index=False)
