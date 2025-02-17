import os
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging
from PIL import Image, ImageDraw
from dataclasses import dataclass
from labeling.label_converter import HardMultiLabelResult, SingleLabelResult

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
            # final_results = self.postprocess_results(folder_path, image_paths, probabilities, labels)
            
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
        
    
    def sliding_window_postprocessing(self, folder_probabilities, window_size=5, step=1,
                                    method='combined', sigma=1.0):
        """
        スライディングウィンドウ後処理により、最終的な予測ラベルを決定する。

        Args:
            folder_probabilities: 各フレームごとの出力確率のリスト（各要素はリストまたは配列）
                                ※ 各要素の長さは、モデル出力の次元（例:7または15）
            window_size: 1ウィンドウあたりのフレーム数
            step: ウィンドウをスライドさせるステップ幅
            method: 'center'（中心フレーム重みのみ）,
                    'soft_label'（信頼度のみ）,
                    'combined'（両方の組み合わせ）のいずれか
            sigma: ガウス重みを計算する際の分散パラメータ（中心重み付け用）

        Returns:
            final_predictions: 各ウィンドウに対する最終予測ラベル（主タスク:0～5 のうちの値）
            final_scores: ウィンドウごとの重み付き平均確率（主タスク部分のみ、長さ6の配列）
        """
        # folder_probabilities を numpy 配列に変換（形状：[num_frames, num_classes]）
        probs_array = np.array(folder_probabilities)  # shape = (N, num_classes)
        num_frames, num_classes = probs_array.shape

        final_predictions = []
        final_scores = []  # 各ウィンドウの重み付き平均確率（主タスク部分のみ）

        # 各ウィンドウについて処理
        for start in range(0, num_frames - window_size + 1, step):
            window = probs_array[start:start+window_size, :]  # shape (window_size, num_classes)
            # --- 時間的重みの計算 ---
            indices = np.arange(window_size)
            center = (window_size - 1) / 2.0
            # ガウス分布を用いた重み付け：中心に近いほど大きい重み
            time_weights = np.exp(-((indices - center) ** 2) / (2 * sigma ** 2))
            
            # --- 信頼度重みの計算 ---
            # 主タスクは必ず出力されるので、先頭6クラスを利用する
            main_probs = window[:, :6]  # shape (window_size, 6)
            # 各フレームでの信頼度として、主タスクの最大確信度を利用
            conf_weights = np.max(main_probs, axis=1)  # shape (window_size,)
            
            # --- 重みの組み合わせ ---
            if method == 'center':
                combined_weights = time_weights
            elif method == 'soft_label':
                combined_weights = conf_weights
            elif method == 'combined':
                combined_weights = time_weights * conf_weights
            else:
                # デフォルトは等重み
                combined_weights = np.ones(window_size)
            
            # --- サブタスク（不鮮明）の重み補正 ---
            # 7クラスの場合: 不鮮明確率はインデックス6
            # 15クラスの場合: 不鮮明確率はインデックス6～14（ここでは最大値を採用）
            if num_classes == 7:
                unclear_probs = window[:, 6]  # shape (window_size,)
            elif num_classes == 15:
                unclear_probs = np.max(window[:, 6:15], axis=1)  # shape (window_size,)
            else:
                # 不鮮明情報がない場合は1（補正なし）
                unclear_probs = np.ones(window_size)
            # サブタスクが高いほど、そのフレームは不鮮明とみなし、重みを減少させる
            subtask_factor = 1 - unclear_probs  # 例：不鮮明確率が0.8なら、重みは0.2倍になる
            combined_weights = combined_weights * subtask_factor

            # --- ウィンドウ内の主タスク確率の重み付き平均 ---
            # ここでは、主タスクの確率（最初の6クラス）をウィンドウごとに統合
            weighted_avg = np.average(main_probs, axis=0, weights=combined_weights)  # shape (6,)

            # 最終予測は、重み付き平均確率の最大値のインデックス
            pred_label = int(np.argmax(weighted_avg))
            final_predictions.append(pred_label)
            final_scores.append(weighted_avg)

        return final_predictions, final_scores


    # --- 例: results 辞書を用いた後処理 ---
    def postprocess_results(self, results, window_size=16, step=1, method='combined', sigma=1.0):
        """
        各フォルダごとに、推論結果（results）からスライディングウィンドウ後処理を実施し、
        最終的な予測ラベルとそのスコアを返す。

        Args:
            results: 辞書形式の結果 { folder_name: (folder_probabilities, folder_labels, folder_image_paths) }
            window_size: ウィンドウサイズ
            step: スライドのステップ幅
            method: 後処理の方法（'center', 'soft_label', 'combined'）
            sigma: 中心重み付け用のパラメータ

        Returns:
            final_results: フォルダ毎の後処理結果の辞書
                        { folder_name: (final_predictions, final_scores) }
        """
        final_results = {}

        for folder_name, (folder_probabilities, folder_labels, folder_image_paths) in results.items():
            predictions, scores = self.sliding_window_postprocessing(
                folder_probabilities,
                window_size=window_size,
                step=step,
                method=method,
                sigma=sigma
            )
            final_results[folder_name] = (predictions, scores)
            print(f"Folder: {folder_name}, Final predictions: {predictions}")

        return final_results
    

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

    def apply_sliding_window_to_hard_labels(self, hard_multilabel_results, window_size=5, step=1):
        """
        スライディングウィンドウを適用して、平滑化されたラベルを生成する関数

        Args:
            hard_multilabel_results (dict[str, HardMultiLabelResult]): 各フォルダのマルチラベルの結果
            window_size (int): スライディングウィンドウのサイズ
            step (int): スライディングウィンドウのステップ幅

        Returns:
            dict[str, SingleLabelResult]: 各フォルダの平滑化されたラベル
        """
        smoothed_results = {}

        for folder_name, result in hard_multilabel_results.items():
            y_pred = np.array(result.multilabels)[:, :6]  # 主クラスのみ
            y_true = np.array(result.ground_truth_labels)[:, :6]  # 主クラスのみ
            num_frames, num_classes = y_pred.shape

            smoothed_labels = []
            smoothed_ground_truth = []

            for start in range(0, num_frames - window_size + 1, step):
                window_pred = y_pred[start:start + window_size]
                window_true = y_true[start:start + window_size]

                class_counts_pred = window_pred.sum(axis=0)
                class_counts_true = window_true.sum(axis=0)

                smoothed_label = np.argmax(class_counts_pred)
                smoothed_true_label = np.argmax(class_counts_true)

                smoothed_labels.append(smoothed_label)
                smoothed_ground_truth.append(smoothed_true_label)

            smoothed_results[folder_name] = SingleLabelResult(
                image_paths=result.image_paths[:len(smoothed_labels)],
                single_labels=smoothed_labels,
                ground_truth_labels=smoothed_ground_truth
            )

        return smoothed_results


###
# 以下メイン関数下で使用するコード
###
def load_raw_results(csv_path, num_classes):
    """
    raw_results.csv から画像パス、予測確率、正解ラベルを抽出する関数。

    Args:
        csv_path (str): raw_results.csv のパス
        num_classes (int): クラス数（例: 15）

    Returns:
        image_paths (list of str): 画像パスのリスト
        probabilities (list of list of float): 各画像ごとの予測確率（長さ num_classes のリスト）
        labels (list of list of int): 各画像ごとの正解ラベル（one-hot形式、長さ num_classes のリスト）
    """
    df = pd.read_csv(csv_path)
    
    # 画像パスの抽出
    image_paths = df['Image_Path'].tolist()
    
    # 予測確率の抽出：列名 "Pred_Class_0"～"Pred_Class_{num_classes-1}"
    pred_cols = [f"Pred_Class_{i}" for i in range(num_classes)]
    probabilities = df[pred_cols].values.tolist()
    
    # 正解ラベルの抽出：列名 "True_Class_0"～"True_Class_{num_classes-1}"
    true_cols = [f"True_Class_{i}" for i in range(num_classes)]
    labels = df[true_cols].values.tolist()
    
    # 正解ラベルを整数型に変換（one-hot形式の場合）
    labels = [list(map(int, row)) for row in labels]
    
    return image_paths, probabilities, labels


def majority_vote(window):
    # 各ラベルクラスの出現回数を合計
    class_counts = window.sum(axis=0)
    # 出現回数が最大のクラスを選択
    majority_label = np.argmax(class_counts)
    return majority_label

# スライディングウィンドウ適用関数
def apply_sliding_window(predicted_labels, window_size):
    smoothed_labels = []
    half_window = window_size // 2

    for i in range(len(predicted_labels)):
        start = max(0, i - half_window)
        end = min(len(predicted_labels), i + half_window + 1)
        window = predicted_labels[start:end]
        smoothed_label = majority_vote(window)
        smoothed_labels.append(smoothed_label)
    
    return np.array(smoothed_labels)


def calculate_metrics(true_labels, smoothed_labels):
    cm = confusion_matrix(true_labels, smoothed_labels, labels=np.arange(6))
    # labels=np.arange(6)で0〜5の6クラスに対して計算
    precision = precision_score(true_labels, smoothed_labels, average=None, labels=np.arange(6), zero_division=0)
    recall = recall_score(true_labels, smoothed_labels, average=None, labels=np.arange(6), zero_division=0)
    
    metrics = []
    for cls in range(6):  # 0〜5の6クラス
        metrics.append({
            "Class": cls,
            "Precision": round(precision[cls], 4),
            "Recall": round(recall[cls], 4)
        })
    return cm, metrics

def visualize_timeline(labels, save_dir, filename, n_class):
    """
    マルチラベルタイムラインを可視化して保存
    """
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

    # Determine the number of images
    n_images = len(labels)
    
    # Set timeline height based on the number of labels
    timeline_width = n_images
    # timeline_height = 2 * (n_images // 10)  # 20 pixels per label row (2 rows total)
    timeline_height = (n_images // 10)  # 20 pixels per label row (1 rows total)

    # Create a blank image for the timeline
    timeline_image = Image.new('RGB', (timeline_width, timeline_height), (255, 255, 255))
    draw = ImageDraw.Draw(timeline_image)

    # Iterate over each image (row in the CSV)
    for i in range(n_images):
        # Get the predicted labels for the current image
        label = labels[i]
        
        # Calculate the position in the timeline
        x1 = i * (timeline_width // n_images)
        x2 = (i + 1) * (timeline_width // n_images)
        y1 = 0
        y2 = (n_images // 10)
        
        # Get the color for the current label
        color = label_colors.get(label, default_color)
        
        # Draw the rectangle for the label
        draw.rectangle([x1, y1, x2, y2], fill=color)
                
    # Save the image
    os.makedirs(save_dir, exist_ok=True)
    timeline_image.save(os.path.join(save_dir, f'{filename}.png'))
    print(f'Timeline image saved at {os.path.join(save_dir, f"{filename}.png")}')
    
    
def process_all_results(dataset_root, num_classes, window_size=5, step=1, methods=None):
    analyzer = Analyzer(dataset_root, num_classes)
    fold_metrics = {}
    all_predictions = []
    all_true_labels = []
    
    for fold in sorted(os.listdir(dataset_root)):
        fold_path = os.path.join(dataset_root, fold)
        if not os.path.isdir(fold_path):
            continue
            
        fold_predictions = []
        fold_true_labels = []
        
        for video_folder in sorted(os.listdir(fold_path)):
            video_path = os.path.join(fold_path, video_folder)
            csv_path = os.path.join(video_path, 'raw_results.csv')
            
            if not os.path.exists(csv_path):
                logging.warning(f'CSV file not found in {video_path}')
                continue
                
            print(f'Processing {csv_path} ...')
            image_paths, probabilities, labels = load_raw_results(csv_path, num_classes)
            
            # Generate predictions
            pred_labels = analyzer.threshold_predictions(probabilities)
            pred_labels_smoothed = apply_sliding_window(pred_labels, window_size=window_size)
            
            # Store for fold-level and overall metrics
            true_labels = [np.argmax(label[:6]) for label in labels]
            fold_predictions.extend(pred_labels_smoothed)
            fold_true_labels.extend(true_labels)
            all_predictions.extend(pred_labels_smoothed)
            all_true_labels.extend(true_labels)
            
            # Calculate and save video-level results
            cm, metrics = calculate_metrics(true_labels, pred_labels_smoothed)
            
            # Save video-level confusion matrix
            save_confusion_matrix(
                cm, 
                os.path.join(video_path, 'confusion_matrix.csv'),
                normalized=True
            )
            
            pd.DataFrame(metrics).to_csv(
                os.path.join(video_path, 'metrics.csv'), 
                index=False
            )
            
            visualize_timeline(
                labels=pred_labels_smoothed,
                save_dir=video_path,
                filename="prediction_timeline",
                n_class=6
            )
        
        # Calculate and save fold-level results
        fold_cm, fold_metrics[fold] = calculate_metrics(fold_true_labels, fold_predictions)
        fold_results_dir = os.path.join(fold_path, 'fold_results')
        os.makedirs(fold_results_dir, exist_ok=True)
        
        # Save fold-level confusion matrix (both raw and normalized)
        save_confusion_matrix(
            fold_cm,
            os.path.join(fold_results_dir, 'fold_confusion_matrix_raw.csv'),
            normalized=False
        )
        save_confusion_matrix(
            fold_cm,
            os.path.join(fold_results_dir, 'fold_confusion_matrix_normalized.csv'),
            normalized=True
        )
        
        pd.DataFrame(fold_metrics[fold]).to_csv(
            os.path.join(fold_results_dir, 'fold_metrics.csv'),
            index=False
        )
        
        visualize_timeline(
            labels=fold_predictions,
            save_dir=fold_results_dir,
            filename="fold_prediction_timeline",
            n_class=6
        )
    
    # Calculate and save overall results
    overall_cm, overall_metrics = calculate_metrics(all_true_labels, all_predictions)
    
    # Save overall confusion matrices
    save_confusion_matrix(
        overall_cm,
        os.path.join(dataset_root, 'overall_confusion_matrix_raw.csv'),
        normalized=False
    )
    save_confusion_matrix(
        overall_cm,
        os.path.join(dataset_root, 'overall_confusion_matrix_normalized.csv'),
        normalized=True
    )
    
    # Save overall metrics summary
    pd.DataFrame(overall_metrics).to_csv(
        os.path.join(dataset_root, 'overall_metrics.csv'),
        index=False
    )
    
    return fold_metrics, overall_metrics

# def process_all_results(dataset_root, num_classes, window_size=5, step=1, methods=None):
#     analyzer = Analyzer(dataset_root, num_classes)
#     fold_metrics = {}
    
#     for fold in sorted(os.listdir(dataset_root)):
#         fold_path = os.path.join(dataset_root, fold)
#         if not os.path.isdir(fold_path):
#             continue
            
#         fold_predictions = []
#         fold_true_labels = []
        
#         for video_folder in sorted(os.listdir(fold_path)):
#             video_path = os.path.join(fold_path, video_folder)
#             csv_path = os.path.join(video_path, 'raw_results.csv')
            
#             if not os.path.exists(csv_path):
#                 logging.warning(f'CSV file not found in {video_path}')
#                 continue
                
#             print(f'Processing {csv_path} ...')
#             image_paths, probabilities, labels = load_raw_results(csv_path, num_classes)
            
#             # Generate predictions
#             pred_labels = analyzer.threshold_predictions(probabilities)
#             pred_labels_smoothed = apply_sliding_window(pred_labels, window_size=window_size)
            
#             # Store for fold-level metrics
#             fold_predictions.extend(pred_labels_smoothed)
#             fold_true_labels.extend([np.argmax(label[:6]) for label in labels])
            
#             # Calculate video-level metrics
#             cm, metrics = calculate_metrics(
#                 [np.argmax(label[:6]) for label in labels], 
#                 pred_labels_smoothed
#             )
            
#             # Save video-level confusion matrix
#             cm_df = pd.DataFrame(
#                 cm, 
#                 index=[f'True_{i}' for i in range(6)], 
#                 columns=[f'Pred_{i}' for i in range(6)]
#             )
#             cm_df.to_csv(os.path.join(video_path, 'confusion_matrix.csv'))
            
#             # Save video-level metrics
#             pd.DataFrame(metrics).to_csv(
#                 os.path.join(video_path, 'metrics.csv'), 
#                 index=False
#             )
            
#             # Generate timeline visualization
#             visualize_timeline(
#                 labels=pred_labels_smoothed,
#                 save_dir=video_path,
#                 filename="prediction_timeline",
#                 n_class=6
#             )
        
#         # Calculate fold-level metrics
#         fold_cm, fold_metrics[fold] = calculate_metrics(
#             fold_true_labels, 
#             fold_predictions
#         )
        
#         # Save fold-level results
#         fold_results_dir = os.path.join(fold_path, 'fold_results')
#         os.makedirs(fold_results_dir, exist_ok=True)
        
#         pd.DataFrame(
#             fold_cm,
#             index=[f'True_{i}' for i in range(6)],
#             columns=[f'Pred_{i}' for i in range(6)]
#         ).to_csv(os.path.join(fold_results_dir, 'fold_confusion_matrix.csv'))
        
#         pd.DataFrame(fold_metrics[fold]).to_csv(
#             os.path.join(fold_results_dir, 'fold_metrics.csv'),
#             index=False
#         )
        
#         visualize_timeline(
#             labels=fold_predictions,
#             save_dir=fold_results_dir,
#             filename="fold_prediction_timeline",
#             n_class=6
#         )
    
#     return fold_metrics
    # """
    # 交差検証のfoldディレクトリ下の各動画フォルダに対してCSVを読み込み、各手法で後処理を実施する。
    
    # ディレクトリ構造例：
    #   dataset_root/
    #       fold1/
    #           VideoFolder1/raw_results.csv
    #           VideoFolder2/raw_results.csv
    #       fold2/
    #           ...
    
    # 結果は辞書形式で返す。
    # """
    
    # # Analyzer クラスのインスタンスを作成
    # analyzer = Analyzer(dataset_root, num_classes)
    
    # all_results = {}
    # for fold in sorted(os.listdir(dataset_root)):
    #     fold_path = os.path.join(dataset_root, fold)
    #     if not os.path.isdir(fold_path):
    #         continue
    #     all_results[fold] = {}
    #     # 各動画フォルダ
    #     for video_folder in sorted(os.listdir(fold_path)):
    #         video_path = os.path.join(fold_path, video_folder)
    #         csv_path = os.path.join(video_path, 'raw_results.csv')
    #         if os.path.exists(csv_path):
    #             print(f'Processing {csv_path} ...')
    #             image_paths, probabilities, labels = load_raw_results(csv_path, num_classes)
    #             ###
    #             # コード部分
    #             ###
    #             # 50%以上閾値で予測ラベル（マルチラベル）を作成後、シンプルなSliding windowで予測ラベル（シングルラベル）を決定
    #             pred_labels = analyzer.threshold_predictions(probabilities)
    #             # print(type(pred_labels))
    #             pred_labels_smoothed = apply_sliding_window(pred_labels, window_size=window_size)
                
                
    #             # break
    #         else:
    #             logging.warning(f'CSV file not found in {video_path}')
    # return all_results    


# ----- 各手法の名称定義 ----- #
# binary_*系は「50%以上閾値」で二値化した後の処理
# prob_*系は確率ベクトルを直接利用
METHODS = {
    1: "binary_simple",
    2: "binary_center",
    3: "prob_simple",
    4: "prob_center",
    5: "prob_entropy",
    6: "prob_center_entropy"
}

# # メトリクス計算・保存
#                 cm, metrics = calculate_metrics(labels, pred_labels_smoothed)
#                 metrics_df = pd.DataFrame(metrics)
                
#                 # 混同行列を保存
#                 cm_df = pd.DataFrame(cm, index=[f'True_{i}' for i in range(6)], columns=[f'Pred_{i}' for i in range(6)])
#                 cm_csv = os.path.join(video_path, f'confusion_matrix_{methods}.csv')
#                 cm_df.to_csv(cm_csv)

#                 metrics_csv = os.path.join(video_path, 'metrics.csv')
#                 metrics_df.to_csv(metrics_csv, index=False)
                
#                 visualize_timeline(
#                     labels=pred_labels_smoothed,
#                     save_dir=video_path,
#                     filename=f"sw_labels",
#                     n_class=6
#                 )
                
#                 # final_predictions, final_scores = analyzer.sliding_window_postprocessing(folder_probabilities, window_size=5, step=1, method='combined', sigma=1.0)
#                 # all_results[fold][video_folder] = result

def save_confusion_matrix(cm, save_path, normalized=False):
    """Save confusion matrix to CSV with optional normalization"""
    if normalized:
        # Normalize by row (true labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.where(row_sums == 0, 0, cm / row_sums)
        cm_to_save = cm_normalized
    else:
        cm_to_save = cm
        
    cm_df = pd.DataFrame(
        cm_to_save,
        index=[f'True_{i}' for i in range(6)],
        columns=[f'Pred_{i}' for i in range(6)]
    )
    cm_df.to_csv(save_path)


def main():
    logging.basicConfig(level=logging.INFO)
    num_classes = 15
    save_dir = f"{num_classes}class_results"
    os.makedirs(save_dir, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO)
    num_classes = 15
    save_dir = f"{num_classes}class_results"
    os.makedirs(save_dir, exist_ok=True)
    
    fold_metrics = process_all_results(
        dataset_root=save_dir,
        num_classes=num_classes,
        window_size=16,
        step=1
    )
    
    # Save overall results summary
    overall_metrics = pd.DataFrame([
        {
            'Fold': fold,
            **{f'Class_{i}_{metric}': metrics[i][metric] 
               for i in range(6) 
               for metric in ['Precision', 'Recall']}
        }
        for fold, metrics in fold_metrics.items()
    ])
    
    overall_metrics.to_csv(
        os.path.join(save_dir, 'overall_results.csv'),
        index=False
    )
    
    # logging.basicConfig(level=logging.INFO)
    
    # # クラス数（例: 15クラスの場合）
    # num_classes = 15
    
    # # 結果を保存するディレクトリ（存在しなければ作成）
    # save_dir = f"{num_classes}class_results"
    # os.makedirs(save_dir, exist_ok=True)
    
    # process_all_results(dataset_root=save_dir, num_classes=num_classes, window_size=16, step=1, methods=METHODS[1])
    
    # ダミーの推論結果（Inference.run() の出力形式に準拠）を作成
    # results = create_dummy_results(num_frames=50, num_classes=num_classes)
    
    # Analyzer.analyze() を呼び出して、以下の処理を実施する
    # ・生の結果のCSV保存 (raw_results.csv)
    # ・50%閾値でのバイナリ予測結果の作成・保存 (threshold_results.csv)
    # ・各クラスごとの適合率、再現率、混同行列、分類レポートの算出 (metrics.csv, confusion_matrix.csv)
    # ・マルチラベルのタイムライン画像と正解タイムライン画像の作成
    # analyzer.analyze(results)
    
    # また、もしスライディングウィンドウ後処理の関数を個別に利用する場合は、
    # 以下のように Analyzer.postprocess_results() を呼び出すこともできます。
    # （メソッド内では Analyzer.sliding_window_postprocessing() を利用しています）
    # window_size = 16
    # step = 1
    # postprocessed_results = analyzer.postprocess_results(results, window_size=window_size, step=step, method='combined', sigma=1.0)
    # logging.info("Postprocessing complete.")
    
    # ここで、postprocessed_results には各フォルダごとに (final_predictions, final_scores) が保存されているので、
    # さらに個別の評価指標やタイムライン画像の生成に利用可能です。
    
if __name__ == '__main__':
    main()

def apply_sliding_window_to_hard_labels(hard_multilabel_results: dict[str, HardMultiLabelResult], window_size: int = 5, step: int = 1) -> dict[str, SingleLabelResult]:
    """
    スライディングウィンドウを適用して、平滑化されたラベルを生成する関数

    Args:
        hard_multilabel_results (dict[str, HardMultiLabelResult]): 各フォルダのマルチラベルの結果
        window_size (int): スライディングウィンドウのサイズ
        step (int): スライディングウィンドウのステップ幅

    Returns:
        dict[str, SingleLabelResult]: 各フォルダの平滑化されたラベル
    """
    smoothed_results = {}

    for folder_name, result in hard_multilabel_results.items():
        y_pred = np.array(result.multilabels)[:, :6]  # 主クラスのみ
        y_true = np.array(result.ground_truth_labels)[:, :6]  # 主クラスのみ
        num_frames, num_classes = y_pred.shape

        smoothed_labels = []
        smoothed_ground_truth = []

        for start in range(0, num_frames - window_size + 1, step):
            window_pred = y_pred[start:start + window_size]
            window_true = y_true[start:start + window_size]

            # 各クラスの出現回数を合計し、最も多いクラスを選択
            class_counts_pred = window_pred.sum(axis=0)
            class_counts_true = window_true.sum(axis=0)

            smoothed_label = np.argmax(class_counts_pred)
            smoothed_true_label = np.argmax(class_counts_true)

            smoothed_labels.append(smoothed_label)
            smoothed_ground_truth.append(smoothed_true_label)

        smoothed_results[folder_name] = SingleLabelResult(
            image_paths=result.image_paths[:len(smoothed_labels)],
            single_labels=smoothed_labels,
            ground_truth_labels=smoothed_ground_truth
        )

    return smoothed_results