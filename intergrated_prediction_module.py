import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from PIL import Image, ImageDraw
import os

class PredictionProcessor:
    def __init__(self, num_classes=6):
        self.num_classes = num_classes
        self.label_colors = {
            0: (254, 195, 195),  # white
            1: (204, 66, 38),    # lugol
            2: (57, 103, 177),   # indigo
            3: (96, 165, 53),    # nbi
            4: (86, 65, 72),     # custom color
            5: (159, 190, 183),  # custom color
        }
        self.default_color = (148, 148, 148)

    def process_max_probability(self, data, prob_columns):
        """確率最大のクラスを予測として選択"""
        return data[prob_columns].idxmax(axis=1).apply(lambda x: int(x.split("_")[1]))

    def apply_sliding_window(self, labels, window_size, method="majority"):
        """スライディングウィンドウによる平滑化"""
        smoothed_labels = np.zeros_like(labels)
        half_window = window_size // 2
        
        for i in range(len(labels)):
            start = max(0, i - half_window)
            end = min(len(labels), i + half_window + 1)
            window = labels[start:end]
            
            if method == "majority":
                smoothed_labels[i] = np.bincount(window).argmax()
                
        return smoothed_labels

    def apply_probability_window(self, prob_data, window_size):
        """確率値に対するスライディングウィンドウ処理"""
        smoothed_probs = np.zeros_like(prob_data)
        half_window = window_size // 2
        
        for i in range(len(prob_data)):
            start = max(0, i - half_window)
            end = min(len(prob_data), i + half_window + 1)
            window = prob_data[start:end]
            smoothed_probs[i] = np.mean(window, axis=0)
            
        return np.argmax(smoothed_probs, axis=1)

    def calculate_metrics(self, true_labels, pred_labels):
        """評価指標の計算"""
        cm = confusion_matrix(true_labels, pred_labels, labels=range(self.num_classes))
        precision = precision_score(true_labels, pred_labels, average=None, 
                                  labels=range(self.num_classes), zero_division=0)
        recall = recall_score(true_labels, pred_labels, average=None, 
                            labels=range(self.num_classes), zero_division=0)
        
        metrics = {
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall
        }
        
        return metrics

    def visualize_timeline(self, labels, save_path):
        """予測結果のタイムライン可視化"""
        n_images = len(labels)
        timeline_width = n_images
        timeline_height = n_images // 10

        timeline_image = Image.new('RGB', (timeline_width, timeline_height), (255, 255, 255))
        draw = ImageDraw.Draw(timeline_image)

        for i in range(n_images):
            label = labels[i]
            x1 = i * (timeline_width // n_images)
            x2 = (i + 1) * (timeline_width // n_images)
            y1 = 0
            y2 = timeline_height
            
            color = self.label_colors.get(label, self.default_color)
            draw.rectangle([x1, y1, x2, y2], fill=color)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        timeline_image.save(save_path)

    def process_and_evaluate(self, input_path, output_dir, method="max_prob", window_size=31):
        """予測処理と評価の実行"""
        os.makedirs(output_dir, exist_ok=True)
        data = pd.read_csv(input_path)
        
        # 予測方法の選択
        if method == "max_prob":
            prob_columns = [f"Class_{i}_Prob" for i in range(self.num_classes)]
            predictions = self.process_max_probability(data, prob_columns)
        elif method == "sliding_max":
            prob_columns = [f"Class_{i}_Prob" for i in range(self.num_classes)]
            initial_preds = self.process_max_probability(data, prob_columns)
            predictions = self.apply_sliding_window(initial_preds, window_size)
        elif method == "prob_window":
            prob_columns = [f"Class_{i}_Prob" for i in range(self.num_classes)]
            prob_data = data[prob_columns].values
            predictions = self.apply_probability_window(prob_data, window_size)
            
        # 真のラベルの取得
        label_columns = [f"Class_{i}_Label" for i in range(self.num_classes)]
        true_labels = data[label_columns].idxmax(axis=1).apply(lambda x: int(x.split("_")[1]))
        
        # 評価指標の計算
        metrics = self.calculate_metrics(true_labels, predictions)
        
        # 結果の保存
        data['Predicted_Label'] = predictions
        data['True_Label'] = true_labels
        data.to_csv(os.path.join(output_dir, f"{method}_predictions.csv"), index=False)
        
        # 混同行列の保存
        cm_df = pd.DataFrame(metrics['confusion_matrix'], 
                           index=[f"True_{i}" for i in range(self.num_classes)],
                           columns=[f"Pred_{i}" for i in range(self.num_classes)])
        cm_df.to_csv(os.path.join(output_dir, f"{method}_confusion_matrix.csv"))
        
        # 評価指標の保存
        metrics_df = pd.DataFrame({
            'Class': range(self.num_classes),
            'Precision': metrics['precision'],
            'Recall': metrics['recall']
        })
        metrics_df.to_csv(os.path.join(output_dir, f"{method}_metrics.csv"), index=False)
        
        # タイムラインの可視化
        self.visualize_timeline(predictions, 
                              os.path.join(output_dir, f"{method}_timeline.png"))
        
        return metrics

def main():
    # 使用例
    processor = PredictionProcessor(num_classes=6)
    
    # 設定
    input_path = "path/to/input.csv"
    output_dir = "results"
    
    # 各手法での処理実行
    methods = ["max_prob", "sliding_max", "prob_window"]
    window_sizes = [31]
    
    for method in methods:
        for window_size in window_sizes:
            result_dir = os.path.join(output_dir, f"{method}_w{window_size}")
            metrics = processor.process_and_evaluate(
                input_path, 
                result_dir,
                method=method,
                window_size=window_size
            )
            print(f"Method: {method}, Window Size: {window_size}")
            print("Metrics:", metrics)

if __name__ == "__main__":
    main()