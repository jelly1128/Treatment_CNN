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


###
# 以下メイン関数下で使用するコード
###
def visualize_timeline(labels, save_dir, filename, num_classes):
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


# def majority_vote(window):
#     # 各ラベルクラスの出現回数を合計
#     class_counts = window.sum(axis=0)
#     # 出現回数が最大のクラスを選択
#     majority_label = np.argmax(class_counts)
#     return majority_label

# def majority_vote(window):
#     """マルチクラス用多数決投票（排他的クラス向け）"""
#     # 各フレームの予測クラス（argmax）を取得
#     pred_classes = np.argmax(window, axis=1)
#     # クラスごとの投票数集計
#     counts = np.bincount(pred_classes, minlength=window.shape[1])
#     # 最多得票クラスを選択（同数の場合は確率の平均で判定）
#     max_votes = np.max(counts)
#     candidates = np.where(counts == max_votes)[0]
    
#     if len(candidates) > 1:
#         # 同数の場合、ウィンドウ内の確率平均が最大のクラスを選択
#         mean_probs = np.mean(window[:, candidates], axis=0)
#         return candidates[np.argmax(mean_probs)]
#     return np.argmax(counts)


def majority_vote(window):
    """マルチクラス用多数決投票（クラスインデックス入力対応版）"""
    # 入力が1次元配列（クラスインデックスのリスト）の場合の処理
    if isinstance(window, np.ndarray) and window.ndim == 1:
        counts = np.bincount(window, minlength=6)  # 0～5クラスを想定
    else:
        counts = np.bincount(window.flatten(), minlength=6)
    
    max_votes = np.max(counts)
    candidates = np.where(counts == max_votes)[0]
    
    if len(candidates) > 1:
        # 同数の場合、ウィンドウの中央の値を採用
        return window[len(window)//2]
    return np.argmax(counts)

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
    
    
# def process_all_results(dataset_root, num_classes, window_size=5, step=1, methods=None):
#     analyzer = Analyzer(dataset_root, num_classes)
#     fold_metrics_raw = {}
#     fold_metrics_smoothed = {}
#     all_predictions_raw = []
#     all_predictions_smoothed = []
#     all_true_labels = []
    
#     for fold in sorted(os.listdir(dataset_root)):
#         fold_path = os.path.join(dataset_root, fold)
#         if not os.path.isdir(fold_path):
#             continue
            
#         fold_predictions_raw = []
#         fold_predictions_smoothed = []
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
#             pred_labels_raw = analyzer.threshold_predictions(probabilities)
#             pred_labels_raw = np.array([np.argmax(label[:6]) for label in pred_labels_raw])
#             pred_labels_smoothed = apply_sliding_window(pred_labels_raw, window_size=window_size)
            
#             true_labels = [np.argmax(label[:6]) for label in labels]
            
#             # Store for fold-level metrics
#             fold_predictions_raw.extend(pred_labels_raw)
#             fold_predictions_smoothed.extend(pred_labels_smoothed)
#             fold_true_labels.extend(true_labels)
            
#             # Calculate and save video-level metrics (raw predictions)
#             cm_raw, metrics_raw = calculate_metrics(true_labels, pred_labels_raw)
#             save_confusion_matrix(cm_raw, os.path.join(video_path, 'confusion_matrix_raw.csv'))
#             pd.DataFrame(metrics_raw).to_csv(os.path.join(video_path, 'metrics_raw.csv'), index=False)
            
#             # Calculate and save video-level metrics (smoothed predictions)
#             cm_smoothed, metrics_smoothed = calculate_metrics(true_labels, pred_labels_smoothed)
#             save_confusion_matrix(cm_smoothed, os.path.join(video_path, 'confusion_matrix_smoothed.csv'))
#             pd.DataFrame(metrics_smoothed).to_csv(os.path.join(video_path, 'metrics_smoothed.csv'), index=False)
        
#         # Store for overall metrics
#         all_predictions_raw.extend(fold_predictions_raw)
#         all_predictions_smoothed.extend(fold_predictions_smoothed)
#         all_true_labels.extend(fold_true_labels)
        
#         # Calculate fold-level metrics
#         fold_results_dir = os.path.join(fold_path, 'fold_results')
#         os.makedirs(fold_results_dir, exist_ok=True)
        
#         # Raw predictions
#         fold_cm_raw, fold_metrics_raw[fold] = calculate_metrics(fold_true_labels, fold_predictions_raw)
#         save_confusion_matrix(fold_cm_raw, os.path.join(fold_results_dir, 'fold_confusion_matrix_raw.csv'))
#         pd.DataFrame(fold_metrics_raw[fold]).to_csv(os.path.join(fold_results_dir, 'fold_metrics_raw.csv'), index=False)
        
#         # Smoothed predictions
#         fold_cm_smoothed, fold_metrics_smoothed[fold] = calculate_metrics(fold_true_labels, fold_predictions_smoothed)
#         save_confusion_matrix(fold_cm_smoothed, os.path.join(fold_results_dir, 'fold_confusion_matrix_smoothed.csv'))
#         pd.DataFrame(fold_metrics_smoothed[fold]).to_csv(os.path.join(fold_results_dir, 'fold_metrics_smoothed.csv'), index=False)
    
#     # Calculate and save overall metrics
#     overall_cm_raw, overall_metrics_raw = calculate_metrics(all_true_labels, all_predictions_raw)
#     overall_cm_smoothed, overall_metrics_smoothed = calculate_metrics(all_true_labels, all_predictions_smoothed)
    
#     save_confusion_matrix(overall_cm_raw, os.path.join(dataset_root, 'overall_confusion_matrix_raw.csv'))
#     save_confusion_matrix(overall_cm_smoothed, os.path.join(dataset_root, 'overall_confusion_matrix_smoothed.csv'))
    
#     return {
#         'raw': fold_metrics_raw,
#         'smoothed': fold_metrics_smoothed,
#         'overall_raw': overall_metrics_raw,
#         'overall_smoothed': overall_metrics_smoothed
#     }


def process_all_results(dataset_root, num_classes, window_sizes=[3,5,7,9,11,13,15], step=1):
    analyzer = Analyzer(dataset_root, num_classes)
    results_by_window = {}
    
    for window_size in window_sizes:
        print(f"\nProcessing window size: {window_size}")
        fold_metrics_raw = {}
        fold_metrics_smoothed = {}
        all_predictions_raw = []
        all_predictions_smoothed = []
        all_true_labels = []
        
        for fold in sorted(os.listdir(dataset_root)):
            fold_path = os.path.join(dataset_root, fold)
            if not os.path.isdir(fold_path):
                continue
                
            fold_predictions_raw = []
            fold_predictions_smoothed = []
            fold_true_labels = []
            
            for video_folder in sorted(os.listdir(fold_path)):
                video_path = os.path.join(fold_path, video_folder)
                csv_path = os.path.join(video_path, 'raw_results.csv')
                
                if not os.path.exists(csv_path):
                    logging.warning(f'CSV file not found in {video_path}')
                    continue
                    
                print(f'Processing {csv_path}')
                image_paths, probabilities, labels = load_raw_results(csv_path, num_classes)
                
                pred_labels_raw = analyzer.threshold_predictions(probabilities)
                pred_labels_raw = np.array([np.argmax(label[:6]) for label in pred_labels_raw])
                pred_labels_smoothed = apply_sliding_window(pred_labels_raw, window_size=window_size)
                
                if window_size == 1:
                    pred_labels_smoothed = [int(np.argmax(probs[:6])) for probs in probabilities]
                
                true_labels = [np.argmax(label[:6]) for label in labels]
                
                # Store for fold-level metrics
                fold_predictions_raw.extend(pred_labels_raw)
                fold_predictions_smoothed.extend(pred_labels_smoothed)
                fold_true_labels.extend(true_labels)
                
                # Save results in window size specific directory
                window_results_dir = os.path.join(video_path, f'window_{window_size}')
                os.makedirs(window_results_dir, exist_ok=True)
                
                # Save video-level metrics
                for pred_type, preds in [('raw', pred_labels_raw), ('smoothed', pred_labels_smoothed)]:
                    cm, metrics = calculate_metrics(true_labels, preds)
                    save_confusion_matrix(cm, os.path.join(window_results_dir, f'confusion_matrix_{pred_type}.csv'))
                    pd.DataFrame(metrics).to_csv(os.path.join(window_results_dir, f'metrics_{pred_type}.csv'), index=False)
                    
                # タイムライン画像生成
                video_name = os.path.basename(video_folder)
                timeline_dir = os.path.join(window_results_dir, 'timelines')
                os.makedirs(timeline_dir, exist_ok=True)

                # 生の予測結果のタイムライン
                visualize_timeline(
                    pred_labels_raw, 
                    timeline_dir, 
                    f'{video_name}_raw_timeline',
                    num_classes
                )

                # 平滑化後の予測結果のタイムライン
                visualize_timeline(
                    pred_labels_smoothed,
                    timeline_dir,
                    f'{video_name}_smoothed_timeline',
                    num_classes
                )
            
            # Save fold-level results
            fold_results_dir = os.path.join(fold_path, f'window_{window_size}/fold_results')
            os.makedirs(fold_results_dir, exist_ok=True)
            
            # Calculate and save fold metrics
            fold_cm_raw, fold_metrics_raw[fold] = calculate_metrics(fold_true_labels, fold_predictions_raw)
            fold_cm_smoothed, fold_metrics_smoothed[fold] = calculate_metrics(fold_true_labels, fold_predictions_smoothed)
            
            save_confusion_matrix(fold_cm_raw, os.path.join(fold_results_dir, 'fold_confusion_matrix_raw.csv'))
            save_confusion_matrix(fold_cm_smoothed, os.path.join(fold_results_dir, 'fold_confusion_matrix_smoothed.csv'))
            
            all_predictions_raw.extend(fold_predictions_raw)
            all_predictions_smoothed.extend(fold_predictions_smoothed)
            all_true_labels.extend(fold_true_labels)
        
        # Calculate overall metrics for this window size
        overall_cm_raw, overall_metrics_raw = calculate_metrics(all_true_labels, all_predictions_raw)
        overall_cm_smoothed, overall_metrics_smoothed = calculate_metrics(all_true_labels, all_predictions_smoothed)
        
        window_dir = os.path.join(dataset_root, f'window_{window_size}')
        os.makedirs(window_dir, exist_ok=True)
        
        save_confusion_matrix(overall_cm_raw, os.path.join(window_dir, 'overall_confusion_matrix_raw.csv'))
        save_confusion_matrix(overall_cm_smoothed, os.path.join(window_dir, 'overall_confusion_matrix_smoothed.csv'))
        
        results_by_window[window_size] = {
            'raw': fold_metrics_raw,
            'smoothed': fold_metrics_smoothed,
            'overall_raw': overall_metrics_raw,
            'overall_smoothed': overall_metrics_smoothed
        }
    
    # Create comprehensive summary across all window sizes
    create_window_size_comparison(results_by_window, dataset_root)
    
    
    return results_by_window

def create_window_size_comparison(results_by_window, save_dir):
    """全window_sizeのクラス別指標を比較するCSVを生成"""
    window_summaries = []
    
    for window_size, results in results_by_window.items():
        # 各クラスの指標を抽出
        for class_idx in range(6):
            raw_metrics = next(m for m in results['overall_raw'] if m['Class'] == class_idx)
            smoothed_metrics = next(m for m in results['overall_smoothed'] if m['Class'] == class_idx)
            
            summary = {
                'window_size': window_size,
                'class': class_idx,
                'raw_precision': raw_metrics['Precision'],
                'raw_recall': raw_metrics['Recall'],
                'smoothed_precision': smoothed_metrics['Precision'],
                'smoothed_recall': smoothed_metrics['Recall'],
                'precision_diff': round(smoothed_metrics['Precision'] - raw_metrics['Precision'], 4),
                'recall_diff': round(smoothed_metrics['Recall'] - raw_metrics['Recall'], 4)
            }
            window_summaries.append(summary)
    
    # DataFrameに変換してソート
    df = pd.DataFrame(window_summaries)
    df = df.sort_values(['class', 'window_size'])
    
    # CSV保存
    comparison_path = os.path.join(save_dir, 'window_size_class_metrics.csv')
    df.to_csv(comparison_path, index=False)
    print(f'Window size comparison saved: {comparison_path}')

# def create_window_size_comparison(results_by_window, save_dir):
#     """Create summary comparing metrics across different window sizes"""
#     window_summaries = []
    
#     for window_size, results in results_by_window.items():
#         # For each class
#         for class_idx in range(6):
#             summary = {
#                 'window_size': window_size,
#                 'class': class_idx,
#                 'raw_precision': results['overall_raw'][class_idx]['Precision'],
#                 'raw_recall': results['overall_raw'][class_idx]['Recall'],
#                 'smoothed_precision': results['overall_smoothed'][class_idx]['Precision'],
#                 'smoothed_recall': results['overall_smoothed'][class_idx]['Recall']
#             }
#             window_summaries.append(summary)
    
#     # Save comprehensive summary
#     pd.DataFrame(window_summaries).to_csv(
#         os.path.join(save_dir, 'window_size_comparison.csv'),
#         index=False
#     )


def save_confusion_matrix(cm, save_path):
    """Save confusion matrix to CSV with both raw and normalized versions"""
    # Raw confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=[f'True_{i}' for i in range(6)],
        columns=[f'Pred_{i}' for i in range(6)]
    )
    cm_df.to_csv(save_path)
    
    # Normalized confusion matrix
    normalized_path = save_path.replace('.csv', '_normalized.csv')
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.where(row_sums == 0, 0, cm / row_sums)
    cm_norm_df = pd.DataFrame(
        cm_normalized,
        index=[f'True_{i}' for i in range(6)],
        columns=[f'Pred_{i}' for i in range(6)]
    )
    cm_norm_df.to_csv(normalized_path)
    

def main():
    logging.basicConfig(level=logging.INFO)
    num_classes = 6
    save_dir = f"{num_classes}class_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # window_sizes = [1,3,5,7, 9,11,13,15,17,19, 21,23, 25,27,29,31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61]
    window_sizes = [1]
    results = process_all_results(
        dataset_root=save_dir,
        num_classes=num_classes,
        window_sizes=window_sizes
    )

# def main():
#     logging.basicConfig(level=logging.INFO)
#     num_classes = 15
#     save_dir = f"{num_classes}class_results"
#     os.makedirs(save_dir, exist_ok=True)
    
#     metrics = process_all_results(
#         dataset_root=save_dir,
#         num_classes=num_classes,
#         window_size=15,
#         step=1
#     )
    
#     # Create summary of all metrics (raw predictions)
#     raw_summary = pd.DataFrame([
#         {
#             'Fold': fold,
#             **{f'Class_{i}_{metric}': fold_metrics[i][metric] 
#                for i in range(6) 
#                for metric in ['Precision', 'Recall']}
#         }
#         for fold, fold_metrics in metrics['raw'].items()
#     ])
    
#     # Add overall results
#     raw_summary = pd.concat([
#         raw_summary,
#         pd.DataFrame([{
#             'Fold': 'Overall',
#             **{f'Class_{i}_{metric}': metrics['overall_raw'][i][metric]
#                for i in range(6)
#                for metric in ['Precision', 'Recall']}
#         }])
#     ])
    
#     raw_summary.to_csv(os.path.join(save_dir, 'metrics_summary_raw.csv'), index=False)
    
#     # Repeat for smoothed predictions
#     smoothed_summary = pd.DataFrame([
#         {
#             'Fold': fold,
#             **{f'Class_{i}_{metric}': fold_metrics[i][metric] 
#                for i in range(6) 
#                for metric in ['Precision', 'Recall']}
#         }
#         for fold, fold_metrics in metrics['smoothed'].items()
#     ])
    
#     smoothed_summary = pd.concat([
#         smoothed_summary,
#         pd.DataFrame([{
#             'Fold': 'Overall',
#             **{f'Class_{i}_{metric}': metrics['overall_smoothed'][i][metric]
#                for i in range(6)
#                for metric in ['Precision', 'Recall']}
#         }])
#     ])
    
#     smoothed_summary.to_csv(os.path.join(save_dir, 'metrics_summary_smoothed.csv'), index=False)

if __name__ == '__main__':
    main()