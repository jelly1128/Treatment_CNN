import csv

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