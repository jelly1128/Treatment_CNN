import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
import os

def process_predictions(input_csv_path, output_dir, prob_column_range=(0, 5)):
    """
    指定された範囲の予測確率列から最も高い確率のクラスを予測ラベルとして計算し、
    結果を新しいCSVファイルに保存する。

    Args:
        input_csv_path (str): 入力CSVファイルのパス。
        output_dir (str): 出力ファイルの保存ディレクトリ。
        prob_column_range (tuple): クラス予測確率列の範囲（開始クラス, 終了クラス）。

    Returns:
        None
    """
    # CSVファイルを読み込む
    data = pd.read_csv(input_csv_path)

    # 指定されたクラスの予測確率の列を抽出
    prob_columns = [f"Class_{i}_Prob" for i in range(prob_column_range[0], prob_column_range[1] + 1)]

    # 最も高い予測確率のクラスを予測ラベルとして計算
    data["Predicted_Label"] = data[prob_columns].idxmax(axis=1).apply(lambda x: int(x.split("_")[1]))
    
    # 正解ラベルの列を取得
    label_columns = [f"Class_{i}_Label" for i in range(prob_column_range[0], prob_column_range[1] + 1)]
    data["True_Label"] = data[label_columns].idxmax(axis=1).apply(lambda x: int(x.split("_")[1]))

    # 必要な列を選択して新しいCSVファイルに保存（正解ラベルも含める）
    output_columns = ["Image_Path", "Predicted_Label", "True_Label"]
    output_csv_path = os.path.join(output_dir, "prob_max_labels.csv")
    data[output_columns].to_csv(output_csv_path, index=False)
    
    
    # 正解ラベル（行ごとに1つのクラス番号に変換）
    data["True_Label"] = data[label_columns].idxmax(axis=1).apply(lambda x: int(x.split("_")[1]))

    # 混同行列を計算
    cm = confusion_matrix(data["True_Label"], data["Predicted_Label"], labels=range(prob_column_range[0], prob_column_range[1] + 1))

    # 適合率・再現率を計算
    report = classification_report(data["True_Label"], data["Predicted_Label"], labels=range(prob_column_range[0], prob_column_range[1] + 1), output_dict=True)

    # 混同行列をCSVに保存
    cm_df = pd.DataFrame(cm, index=[f"Class_{i}" for i in range(prob_column_range[0], prob_column_range[1] + 1)], 
                         columns=[f"Class_{i}" for i in range(prob_column_range[0], prob_column_range[1] + 1)])
    cm_csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path)
    print(f"混同行列を保存しました: {cm_csv_path}")

    # 適合率・再現率をCSVに保存
    report_df = pd.DataFrame(report).transpose()
    report_csv_path = os.path.join(output_dir, "classification_report.csv")
    report_df.to_csv(report_csv_path)
    print(f"適合率・再現率レポートを保存しました: {report_csv_path}")
    
    # タイムラインの可視化
    visualize_timeline(
        labels=data["Predicted_Label"].values,
        save_dir=output_dir,
        filename="pred_max_labels",
        n_class=prob_column_range[1] - prob_column_range[0] + 1
    )
    
    visualize_timeline(
        labels=data["True_Label"].values,
        save_dir=output_dir,
        filename="ground_truth",
        n_class=prob_column_range[1] - prob_column_range[0] + 1
    )
    
    return cm
        
    
# タイムラインの可視化関数
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

# 関数の使用例
# input_csv = "15class/results/15class_fold1/20230803-110626-ES06_20230803-111315-es06-hd/multilabels_test_with_labels.csv"
# output_csv = "debug_output.csv"
# process_predictions(input_csv, output_csv, prob_column_range=(0, 5))


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


def main():
    # folder = "6class/results"
    # class_num = 6
    # output_dir = "debug"
    
    folder = "results"
    class_num = 6
    output_dir = "prob_max_6class"
    # 各テストフォルダ毎の結果の保存folderを作成
    if not os.path.exists(os.path.join(output_dir)):
        os.mkdir(os.path.join(output_dir))
        
    all_cm = 0
    
    for fold in range(4):
        fold_results = {
            'cm': np.zeros((6, 6), dtype=int),  # 混同行列の集約
            'report': {label: 0 for label in range(6)}   # 適合率・再現率の集約
        }
        subfolder_names = [name for name in os.listdir(os.path.join(folder, f"{class_num}class_fold{fold + 1}")) if os.path.isdir(os.path.join(folder, f"{class_num}class_fold{fold + 1}", name))]
        print(subfolder_names)
        # 各テストフォルダ毎の結果の保存folderを作成
        if not os.path.exists(os.path.join(output_dir, f"{class_num}class_fold{fold + 1}")):
            os.mkdir(os.path.join(output_dir, f"{class_num}class_fold{fold + 1}"))
            
        subfolders_cm = 0
            
        for subfolder_name in subfolder_names:
            # 各テストフォルダ毎の結果の保存folderを作成
            if not os.path.exists(os.path.join(output_dir, f"{class_num}class_fold{fold + 1}", subfolder_name)):
                os.mkdir(os.path.join(output_dir, f"{class_num}class_fold{fold + 1}", subfolder_name))
                
            cm = process_predictions(os.path.join(folder, f"{class_num}class_fold{fold + 1}", subfolder_name, "multilabels_test_with_labels.csv"),    # input_csv
                                    os.path.join(output_dir, f"{class_num}class_fold{fold + 1}", subfolder_name), 
                                    prob_column_range=(0, 5)
                                    )
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