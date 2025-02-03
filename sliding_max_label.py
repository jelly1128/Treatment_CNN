import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import os
from PIL import Image, ImageDraw

# 多数決法
def majority_vote(window_labels):
    center_index = len(window_labels) // 2
    label_counts = np.bincount(window_labels)
    majority_label = np.argmax(label_counts)
    return majority_label

# スライディングウィンドウ適用関数
def apply_sliding_window(data, method, window_size):
    predicted_labels = data['Predicted_Label'].to_numpy()
    smoothed_labels = np.zeros_like(predicted_labels)
    
    half_window = window_size // 2
    for i in range(len(predicted_labels)):
        start = max(0, i - half_window)
        end = min(len(predicted_labels), i + half_window + 1)
        window = predicted_labels[start:end]

        if method == "majority_vote":
            smoothed_labels[i] = majority_vote(window)
        # 他の手法も必要に応じて追加可能
    return smoothed_labels

# # メトリクス計算
# def calculate_metrics(true_labels, smoothed_predictions):
#     metrics = []
#     for cls in range(6):  # 0〜5の6クラス固定
#         true = (true_labels == cls).astype(int)
#         pred = (smoothed_predictions == cls).astype(int)
        
#         # 混同行列の計算
#         cm = confusion_matrix(true, pred)
#         precision = precision_score(true, pred, zero_division=0)
#         recall = recall_score(true, pred, zero_division=0)
        
#         metrics.append({
#             "Class": cls,
#             "Confusion_Matrix": cm.tolist(),  # 2D配列をリストに変換
#             "Precision": round(precision, 4),
#             "Recall": round(recall, 4)
#         })
#     return metrics

# # メイン処理の改善
# def process_and_evaluate(input_csv, output_dir, window_size=21):
#     data = pd.read_csv(input_csv)
#     true_labels = data["True_Label"].to_numpy()
    
#     methods = ["majority_vote"]
#     for method in methods:
#         smoothed_predictions = apply_sliding_window(data, method, window_size)
#         data["Smoothed_Label"] = smoothed_predictions
        
#         # 結果を保存
#         output_csv = os.path.join(output_dir, f"{method}_predictions.csv")
#         data.to_csv(output_csv, index=False)

#         # メトリクス計算・保存
#         metrics = calculate_metrics(true_labels, smoothed_predictions)
#         metrics_df = pd.DataFrame(metrics)
#         metrics_csv = os.path.join(output_dir, f"{method}_metrics.csv")
#         metrics_df.to_csv(metrics_csv, index=False)

#         print(f"手法: {method}")
#         print(f"予測結果: {output_csv}")
#         print(f"評価結果: {metrics_csv}")
        
#     visualize_timeline(
#         labels=data["Smoothed_Label"].values,
#         save_dir=output_dir,
#         filename=f"{method}_pred_max_labels",
#         num_classes=6
#     )
    
# メトリクス計算
# def calculate_metrics(true_labels, smoothed_predictions):
#     overall_cm = np.zeros((6, 6), dtype=int)  # 6x6の混同行列を初期化
#     metrics = []
    
#     # 各クラスの混同行列、適合率、再現率を計算
#     for cls in range(6):  # 0〜5の6クラス固定
#         true = (true_labels == cls).astype(int)
#         pred = (smoothed_predictions == cls).astype(int)
        
#         # 混同行列の計算
#         cm = confusion_matrix(true, pred, labels=[0, 1, 2, 3, 4, 5])
#         overall_cm += cm  # 全体の混同行列に追加
        
#         precision = precision_score(true, pred, zero_division=0)
#         recall = recall_score(true, pred, zero_division=0)
        
#         metrics.append({
#             "Class": cls,
#             "Confusion_Matrix": cm.tolist(),  # 2D配列をリストに変換
#             "Precision": round(precision, 4),
#             "Recall": round(recall, 4)
#         })
    
#     return overall_cm, metrics

# メトリクス計算
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

# メイン処理の改善
def process_and_evaluate(input_csv, output_dir, window_size=31):
    data = pd.read_csv(input_csv)
    true_labels = data["True_Label"].to_numpy()
    
    methods = ["majority_vote"]
    for method in methods:
        smoothed_predictions = apply_sliding_window(data, method, window_size)
        data["Smoothed_Label"] = smoothed_predictions
        
        # 結果を保存
        output_csv = os.path.join(output_dir, f"{method}_predictions.csv")
        data.to_csv(output_csv, index=False)

        # メトリクス計算・保存
        overall_cm, metrics = calculate_metrics(true_labels, smoothed_predictions)
        metrics_df = pd.DataFrame(metrics)
        metrics_csv = os.path.join(output_dir, f"{method}_metrics.csv")
        metrics_df.to_csv(metrics_csv, index=False)

        # 全体の混同行列をCSVとして保存
        cm_df = pd.DataFrame(overall_cm, columns=[f"Class_{i}" for i in range(6)], index=[f"Class_{i}" for i in range(6)])
        cm_csv = os.path.join(output_dir, f"{method}_overall_confusion_matrix.csv")
        cm_df.to_csv(cm_csv)

        print(f"手法: {method}")
        print(f"予測結果: {output_csv}")
        print(f"評価結果: {metrics_csv}")
        print(f"全体混同行列: {cm_csv}")
        
    visualize_timeline(
        labels=data["Smoothed_Label"].values,
        save_dir=output_dir,
        filename=f"{method}_pred_max_labels",
        num_classes=6
    )
    
    return overall_cm
    
# タイムラインの可視化関数
def visualize_timeline(labels, save_dir, filename, num_classes):
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

# 使用例
# input_csv = "6class/results/6class_fold1/20230803-110626-ES06_20230803-111315-es06-hd/multilabels_test_with_labels.csv"  # 入力ファイル
# output_dir = "debug"  # 出力ディレクトリ
# process_and_evaluate(input_csv, output_dir)

def calculate_precision_recall(cm):
    """
    混同行列から適合率と再現率を計算
    
    Parameters:
    cm (numpy.ndarray): 混同行列
    
    Returns:
    dict: 各クラスの適合率と再現率
    """
    precision = {}
    recall = {}
    
    for i in range(cm.shape[0]):
        # 適合率: 予測した中で正解の割合
        precision[i] = cm[i, i] / (cm[:, i].sum() + 1e-10)
        
        # 再現率: 実際のクラスの中で正解の割合
        recall[i] = cm[i, i] / (cm[i, :].sum() + 1e-10)
    
    return {
        'precision': precision,
        'recall': recall
    }

# 定数
# WINDOW_SIZE = 21  # スライディングウィンドウのサイズ

def main():
    # folder = "6class/results"
    # class_num = 6
    # output_dir = "debug"
    
    folder = "prob_max_6class"
    class_num = 6
    window_size = 16
    output_dir = f"prob_max_{class_num}class_sw{window_size}"
    # 各テストフォルダ毎の結果の保存folderを作成
    if not os.path.exists(os.path.join(output_dir)):
        os.mkdir(os.path.join(output_dir))
        
    all_cm = 0
    
    for fold in range(4):
        fold_results = {
            'cm': np.zeros((6, 6), dtype=int),  # 混同行列の集約
            'report': {label: 0 for label in range(6)}   # 適合率・再現率の集約
        }
        subfolder_path = os.path.join(folder, f"{class_num}class_fold{fold + 1}")
        subfolder_names = [name for name in os.listdir(subfolder_path) if os.path.isdir(os.path.join(subfolder_path, name))]
        print(subfolder_names)
        # 各テストフォルダ毎の結果の保存folderを作成
        if not os.path.exists(os.path.join(output_dir, f"{class_num}class_fold{fold + 1}")):
            os.mkdir(os.path.join(output_dir, f"{class_num}class_fold{fold + 1}"))
        
        subfolders_cm = 0
        
        for subfolder_name in subfolder_names:
            subfolder_output_dir = os.path.join(output_dir, f"{class_num}class_fold{fold + 1}", subfolder_name)
            input_csv = os.path.join(subfolder_path, subfolder_name, "prob_max_labels.csv")
            # 各テストフォルダ毎の結果の保存folderを作成
            if not os.path.exists(os.path.join(output_dir, f"{class_num}class_fold{fold + 1}", subfolder_name)):
                os.mkdir(os.path.join(output_dir, f"{class_num}class_fold{fold + 1}", subfolder_name))
                
            cm = process_and_evaluate(input_csv, subfolder_output_dir, window_size)
            subfolders_cm += cm
        
        subfolders_cm_df = pd.DataFrame(subfolders_cm, index=[f"Class_{i}" for i in range(0, 6)], 
                            columns=[f"Class_{i}" for i in range(0, 6)])
        cm_csv_path = os.path.join(os.path.join(output_dir, f"{class_num}class_fold{fold + 1}"), "confusion_matrix.csv")
        subfolders_cm_df.to_csv(cm_csv_path)
        print(f"混同行列を保存しました: {cm_csv_path}")
        
        # 適合率と再現率の計算
        metrics = calculate_precision_recall(subfolders_cm)
        
        # メトリクスをDataFrameに変換
        metrics_df = pd.DataFrame({
            'Precision': [metrics['precision'][i] for i in range(6)],
            'Recall': [metrics['recall'][i] for i in range(6)]
        }, index=[f"Class_{i}" for i in range(6)])
        
        # メトリクスをCSVに保存
        metrics_csv_path = os.path.join(os.path.join(output_dir, f"{class_num}class_fold{fold + 1}"), "precision_recall.csv")
        metrics_df.to_csv(metrics_csv_path)
        print(f"適合率と再現率を保存しました: {metrics_csv_path}")
        
        all_cm += subfolders_cm
    
    all_cm_df = pd.DataFrame(all_cm, index=[f"Class_{i}" for i in range(0, 6)], 
                        columns=[f"Class_{i}" for i in range(0, 6)])
    cm_csv_path = os.path.join(os.path.join(output_dir), "confusion_matrix.csv")
    all_cm_df.to_csv(cm_csv_path)
    print(f"混同行列を保存しました: {cm_csv_path}")
    
    # 適合率と再現率の計算
    metrics = calculate_precision_recall(all_cm)
    
    # メトリクスをDataFrameに変換
    metrics_df = pd.DataFrame({
        'Precision': [metrics['precision'][i] for i in range(6)],
        'Recall': [metrics['recall'][i] for i in range(6)]
    }, index=[f"Class_{i}" for i in range(6)])
    
    # メトリクスをCSVに保存
    metrics_csv_path = os.path.join(os.path.join(output_dir), "precision_recall.csv")
    metrics_df.to_csv(metrics_csv_path)
    print(f"適合率と再現率を保存しました: {metrics_csv_path}")
            

if __name__ == '__main__':
    main()