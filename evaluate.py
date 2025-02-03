import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from confplot import plot_confusion_matrix_from_data, plot_confusion_matrix_from_matrix
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import csv
from typing import List, Tuple, Dict
import os
from PIL import Image, ImageDraw, ImageFont

class ModelEvaluator:
    def __init__(self, results_path: str):
        """
        モデル評価クラスの初期化
        
        :param results_path: 評価結果を保存するディレクトリパス
        """
        self.results_path = results_path
        
    def plot_learning_curve(self, loss_history: Dict):
        """
        トレーニングロスとバリデーションロスの学習曲線を描画する関数
        
        Parameters:
        train_losses (list): エポックごとのトレーニングロス
        val_losses (list): エポックごとのバリデーションロス
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(loss_history['train']) + 1), loss_history['train'], label='Training Loss')
        plt.plot(range(1, len(loss_history['val']) + 1), loss_history['val'], label='Validation Loss')
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(f"{self.results_path}/learning_curve.png")
        plt.close()
        
        print(f"Learning curve saved to {self.results_path}/learning_curve.png")
        
    def save_loss_to_csv(self, loss_history: Dict):
        """
        トレーニングロスとバリデーションロスのログをCSVファイルに保存する関数
        
        Parameters:
        loss_history (dict): 各エポックのトレーニングロスとバリデーションロスの履歴
        """
        # csvファイルへの書き込み
        with open(f"{self.results_path}/loss_log.csv", "w") as fp:
            fp.write("epoch,training_loss,validation_loss\n")
            for epoch, (train_loss, val_loss) in enumerate(zip(loss_history['train'], loss_history['val']), start=1):
                fp.write(f"{epoch},{train_loss},{val_loss}\n")
        
        print(f"loss saved to {self.results_path}/loss_log.csv")
        
    def calculate_metrics(self, ground_truths: list, predictions: list, cm: np.ndarray, class_names: list[str]) -> tuple[list[float], list[float], list[float]]:
        """
        クラスごとの精度指標と全体の精度指標を計算する
        
        Parameters:
        ground_truths (list): 正解ラベル
        predictions (list): modelの予測結果
        cm (np.ndarray): 混同行列
        class_names (List[str]): クラス名のリスト

        Returns:
            tuple[list[float], list[float], list[float]]: 適合率、再現率、F1スコア
        """
        report = classification_report(ground_truths, predictions, output_dict=True, zero_division='warn')
        # print(report)
        
        # accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for i in class_names:
            if str(i) in report:
                precisions.append(report[str(i)]['precision'])
                recalls.append(report[str(i)]['recall'])
                f1_scores.append(report[str(i)]['f1-score'])
            else:
                precisions.append(0)
                recalls.append(0)
                f1_scores.append(0)
            
        # return accuracy, precision, recall, f1
        return precisions, recalls, f1_scores
    
    def save_confusion_matrix_to_csv(self, ground_truths: list, predictions: list, cm: np.ndarray, class_names: list[str], mode='detailed'):
        """
        混同行列をCSVファイルに保存する
        
        Parameters:
        ground_truths (list): 正解ラベル
        predictions (list): modelの予測結果
        cm (np.ndarray): 混同行列
        class_names (List[str]): クラス名のリスト
        """
        # 各指標の算出
        # accuracies, precisions, recalls, f1_scores = self.calculate_metrics(ground_truths, predictions)
        precisions, recalls, f1_scores = self.calculate_metrics(ground_truths, predictions, cm, class_names)
        
        with open(f"{self.results_path}/confusion_matrix_{mode}.csv", 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            # ヘッダー行を書き込む
            header = ['True\Predicted'] + class_names + ['Total']
            csvwriter.writerow(header)

            # データ行を書き込む
            row_totals = []
            for i, row in enumerate(cm):
                row_total = sum(row)
                row_data = [class_names[i]] + list(row) + [row_total]
                csvwriter.writerow(row_data)
                row_totals.append(row_total)

            # 列と全ての合計を書き込む
            col_totals = cm.sum(axis=0)
            total = sum(row_totals)
            csvwriter.writerow(['Total'] + list(col_totals) + [total])
            
            # 空白行を書き込む
            csvwriter.writerow([])
            
            # print(len(precisions))
            # print(len(recalls))
            # print(len(f1_scores))
            # print(len(class_names))

            # 各クラスの指標を書き込む
            # csvwriter.writerow([''] + ['Accuracy', 'Precision', 'Recall', 'F1-score'])
            csvwriter.writerow([''] + ['Precision', 'Recall', 'F1-score'])
            for i, class_name in enumerate(class_names):
                csvwriter.writerow([class_name] + [f'{precisions[i]:.4f}', f'{recalls[i]:.4f}', f'{f1_scores[i]:.4f}'])

    def create_confusion_matrix(self, ground_truths: list, predictions: list, n_class: int, mode='detailed'):
        """
        混同行列を出力する関数
        
        Parameters:
        ground_truths (list): 正解ラベル
        predictions (list): modelの予測結果
        n_class (int): クラス数
        mode: 'detailed' (Nクラス) または 'binary' (正常/異常)
        """
        if mode == 'detailed':
            # 15クラスの混同行列の生成
            class_names = [str(i) for i in range(n_class)]
            cm = confusion_matrix(ground_truths, predictions, labels=range(n_class))
        
        elif mode == 'binary':
            # 正常と異常の2クラスに変換後，混同行列の生成
            class_names=['Normal', 'Anomalous']
            binary_class = 3    # 正常クラスの範囲
            ground_truths = ['Normal' if y <= binary_class else 'Anomalous' for y in ground_truths]
            predictions = ['Normal' if y <= binary_class else 'Anomalous' for y in predictions]
            cm = confusion_matrix(ground_truths, predictions, labels=class_names)
            
        elif mode == '5-class':
            # 正常4クラスと異常クラスの合計5クラスの混同行列の生成
            class_names = ['0', '1', '2', '3', 'Anomalous']
            def classify(y):
                if y <= 3:
                    return str(y)
                else:
                    return 'Anomalous'
            ground_truths = [classify(y) for y in ground_truths]
            predictions = [classify(y) for y in predictions]
            cm = confusion_matrix(ground_truths, predictions, labels=class_names)
        
        else:
            raise ValueError("Invalid mode. Choose 'detailed' or 'binary'.")
        
        # ヒートマップを適用
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names,
                        yticklabels=class_names,
                        cbar=False)
        plt.title(f"{mode} confusion matrix")
        plt.xlabel('Prediction')
        plt.ylabel('Ground truth')
        plt.savefig(f"{self.results_path}/confusion_matrix_{mode}.png")
        plt.close()
        print(f"save {self.results_path}/confusion_matrix_{mode}.png")
        
        self.save_confusion_matrix_to_csv(ground_truths, predictions, cm, class_names, mode)
        print(f"save {self.results_path}/confusion_matrix_{mode}.csv")
        
        # 適合率とか再現率とか算出しようとした
        # 行と列の合計値を計算
        # row_sums = cm.sum(axis=1)
        # col_sums = cm.sum(axis=0)
        # total_sum = cm.sum()
        # 混同行列を1行1列拡張
        # cm_padded = np.pad(cm, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        # cm_padded[-1, -1] = total_sum
        # cm_padded[:-1, -1] = row_sums
        # cm_padded[-1, :-1] = col_sums
        # ax = sns.heatmap(cm_padded, annot=True, fmt='d', cmap='Blues', 
        #                 xticklabels=class_names + ['Total'],
        #                 yticklabels=class_names + ['Total'],
        #                 cbar=False)
        
        # print(cm_padded)
        
        # df_cm = pd.DataFrame(cm, index=range(0, len(class_names)), columns=range(0, len(class_names)))
        # print(df_cm)
        # plot_confusion_matrix_from_matrix(df_cm, cmap='PuRd', fz=fz, figsize = [14, 14], show_null_values=0, outputfile=f"{self.results_path}/confusion_matrix_{mode}_confplot.png")
        # plt.close()
        
    def visualize_timeline(self, results_csv_file: str):
        """
        結果をタイムライン状に可視化する関数
        
        Parameters:
        results_csv_file (str): 結果が保存されたcsvファイル
        """ 
        # ラベルとその色のlist
        label_colors = [
            (254, 195, 195),       # white
            (204, 66, 38),         # lugol
            (57, 103, 177),        # indigo
            (96, 165, 53),         # nbi
        ]

        # ラベルと時間のリスト
        labels = []

        # CSVファイルを読み込む
        df = pd.read_csv(results_csv_file)

        # リストに変換
        first_column = df.iloc[:, 1].tolist()  # pred_label が3列目にある
        labels = [int(x) for x in first_column]
        print(len(labels))

        value_counts = df.iloc[:, 2].value_counts()

        # 結果の出力
        # print(value_counts)

        # タイムラインの可視化
        timeline_width = len(labels)
        timeline_height = len(labels) // 10
        timeline_image = Image.new('RGB', (timeline_width, timeline_height), (255, 255, 255))
        draw = ImageDraw.Draw(timeline_image)
        font = ImageFont.truetype('arial.ttf', size=16)

        # ラベルを時系列に並べて表示
        for i, (label) in enumerate(labels):
            # ラベルを表示する位置を計算
            x1 = i * (timeline_width // len(labels))
            x2 = (i + 1) * (timeline_width // len(labels)) - 1
            y1 = 0
            y2 = timeline_height
            # ラベルを色分けして表示
            if label < 4:
                draw.rectangle((x1, y1, x2, y2), fill=label_colors[label], outline=None)
            else:
                draw.rectangle((x1, y1, x2, y2), fill=(148, 148, 148), outline=None)
                
        timeline_image.save(f"{self.results_path}/timeline.png")
        print('Timeline image saved as timeline.png')

    def save_anomaly_test_results(self, image_paths:list, ground_truths: list, predictions: list, probabilities: list):
        """
        異常画像検出テスト結果を出力する関数
        
        Parameters:
        ground_truths (list): 正解ラベル
        predictions (list): modelの予測結果
        probabilities (list): 各クラスの予測確率のリスト
        """     
        with open(f"{self.results_path}/anomaly_test_results.csv", 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            header = ['Image Path', 'Ground Truths', 'Predicted Label'] + [f'Top{i+1}_Label' for i in range(3)] + [f'Top{i+1}_Prob' for i in range(3)]  # top3
            csvwriter.writerow(header)
            
            for img_path, ground_truth, prediction, probability in zip(image_paths, ground_truths, predictions, probabilities):
                top3_indices = probability.argsort()[-3:]
                top3_labels = [i for i in top3_indices]
                formatted_probability = [f"{probability[i]:.3f}" for i in top3_indices]
                row = [img_path, ground_truth, prediction] + top3_labels + formatted_probability
                csvwriter.writerow(row)
        
        print(f'Test results saved to {self.results_path}/anomaly_test_results.csv')
        
        # 結果の可視化
        self.visualize_timeline(f"{self.results_path}/anomaly_test_results.csv")
        
        
    def save_anomaly_test_results_to_csv(self, image_paths:list, ground_truths: list, predictions: list, probabilities: list):
        """
        異常画像検出テスト結果を出力する関数
        
        Parameters:
        ground_truths (list): 正解ラベル
        predictions (list): modelの予測結果
        probabilities (list): 各クラスの予測確率のリスト
        """     
        with open(f"{self.results_path}/anomaly_test_results.csv", 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            header = ['Image Path', 'Ground Truths', 'Predicted Label'] + [f'Top{i+1}_Label' for i in range(3)] + [f'Top{i+1}_Prob' for i in range(3)]  # top3
            csvwriter.writerow(header)
            
            for img_path, ground_truth, prediction, probability in zip(image_paths, ground_truths, predictions, probabilities):
                top3_indices = probability.argsort()[-3:]
                top3_labels = [i for i in top3_indices]
                formatted_probability = [f"{probability[i]:.3f}" for i in top3_indices]
                row = [img_path, ground_truth, prediction] + top3_labels + formatted_probability
                csvwriter.writerow(row)
        
        print(f'Test results saved to {self.results_path}/anomaly_test_results.csv')
        
        
    def save_treatment_test_results(self, image_paths:list, ground_truths: list, predictions: list, probabilities: list, anomaly_test_csv_path: str):
        """
        処置分類テスト結果を出力する関数
        
        Parameters:
        ground_truths (list): 正解ラベル
        predictions (list): modelの予測結果
        probabilities (list): 各クラスの予測確率のリスト
        """
        anomaly_test_results = {
            'image_paths': [],
            'ground_truths': [],
            'predictions': [],
            'top3_labels': [],
            'top3_probability': []
        }
        
        treatment_test_results = {
            'image_paths': image_paths,
            'ground_truths': ground_truths,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        with open(anomaly_test_csv_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # ヘッダー行をスキップ
            
            for row in csvreader:
                anomaly_test_results['image_paths'].append(row[0])
                anomaly_test_results['ground_truths'].append(int(row[1]))
                anomaly_test_results['predictions'].append(int(row[2]))
                anomaly_test_results['top3_labels'].append([int(label) for label in row[3:6]])
                anomaly_test_results['top3_probability'].append([f"{float(probability):.3f}" for probability in row[3:6]])
        
             
        with open(f"{self.results_path}/treatment_test_results.csv", 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            header = ['Image Path', 'Ground Truths', 'Anomaly Predicted Label', 'Treatment Predicted Label']
            csvwriter.writerow(header)
            
            for idx in range(len(anomaly_test_results['image_paths'])):
                if anomaly_test_results['image_paths'][idx] in treatment_test_results['image_paths']:
                    treatment_idx = treatment_test_results['image_paths'].index(anomaly_test_results['image_paths'][idx])
                    treatment_prediction = treatment_test_results['predictions'][treatment_idx]
                    row = [anomaly_test_results['image_paths'][idx], 
                        anomaly_test_results['ground_truths'][idx],
                        anomaly_test_results['predictions'][idx],
                        treatment_prediction]
                else:
                    row = [anomaly_test_results['image_paths'][idx], 
                        anomaly_test_results['ground_truths'][idx],
                        anomaly_test_results['predictions'][idx],
                        anomaly_test_results['predictions'][idx]]
                csvwriter.writerow(row)
        
        # with open(f"{self.results_path}/treatment_test_results.csv", 'w', newline='') as csvfile:
        #     csvwriter = csv.writer(csvfile)
        #     header = ['Image Path', 'Ground Truths', 'Anomaly Predicted Label'] + [f'Prob_Class_{i}' for i in range(3)]
        #     csvwriter.writerow(header)
            
        #     for idx in range(len(anomaly_test_results['image_paths'])):
        #         if anomaly_test_results['image_paths'][idx] in treatment_test_results['image_paths']:
        #             treatment_idx = treatment_test_results['image_paths'].index(anomaly_test_results['image_paths'][idx])
        #             treatment_top3_indices = treatment_test_results['probabilities'][treatment_idx].argsort()[-3:]
        #             treatment_top3_labels = [i for i in treatment_top3_indices]
        #             treatment_formatted_probability = [f"{treatment_test_results['probabilities'][treatment_idx][i]:.3f}" for i in treatment_top3_indices]
        #             row = [anomaly_test_results['image_paths'][idx], 
        #                    anomaly_test_results['ground_truths'][idx],
        #                    treatment_test_results['predictions'][treatment_test_results['image_paths'].index(anomaly_test_results['image_paths'][idx])],
        #                    ] + \
        #                 anomaly_test_results['top3_labels'][idx] + \
        #                 anomaly_test_results['top3_probability'][idx] + \
        #                 treatment_top3_labels + \
        #                 treatment_formatted_probability
        #         else:
        #             row = [anomaly_test_results['image_paths'][idx], 
        #                    anomaly_test_results['ground_truths'][idx],
        #                    anomaly_test_results['predictions'][idx],
        #                    ] + \
        #                 anomaly_test_results['top3_labels'][idx] + \
        #                 anomaly_test_results['top3_probability'][idx]
        #         csvwriter.writerow(row)
        
        print(f'Test results saved to {self.results_path}/treatment_test_results.csv')
        # 結果の可視化
        self.visualize_timeline(f"{self.results_path}/treatment_test_results.csv")
        
        ###############
        ### 混同行列 ###
        ############### 
        treatment_test_results = {
            'image_paths': [],
            'ground_truths': [],
            'anomaly_predictions': [],
            'treatment_predictions': []
        }

        with open(f"{self.results_path}/treatment_test_results.csv", 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # ヘッダー行をスキップ
                
            for row in csvreader:
                treatment_test_results['image_paths'].append(row[0])
                treatment_test_results['ground_truths'].append(int(row[1]))
                treatment_test_results['anomaly_predictions'].append(int(row[2]))
                treatment_test_results['treatment_predictions'].append(int(row[3]))
                
        print(len(treatment_test_results['ground_truths']))
        print(len(treatment_test_results['treatment_predictions']))

        # 15クラスの詳細な混同行列
        self.create_confusion_matrix(treatment_test_results['ground_truths'], treatment_test_results['treatment_predictions'], 15, mode='detailed')
        # 5クラスの混同行列
        self.create_confusion_matrix(treatment_test_results['ground_truths'], treatment_test_results['treatment_predictions'], 15, mode='5-class')
        
        
        ###############
        with open(f"{self.results_path}/hogehoge_results.csv", 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            header = ['Image Path', 'Ground Truths', 'Predicted Label'] + [f'Top{i+1}_Label' for i in range(3)] + [f'Top{i+1}_Prob' for i in range(3)]  # top3
            csvwriter.writerow(header)
            
            for img_path, ground_truth, prediction, probability in zip(image_paths, ground_truths, predictions, probabilities):
                top3_indices = probability.argsort()[-3:]
                top3_labels = [i for i in top3_indices]
                formatted_probability = [f"{probability[i]:.3f}" for i in top3_indices]
                row = [img_path, ground_truth, prediction] + top3_labels + formatted_probability
                csvwriter.writerow(row)
        
        
def main():
    # 使用例
    evaluator = ModelEvaluator('anomaly(treatment)_test_4class')
    # train_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
    # val_losses = [0.55, 0.45, 0.35, 0.25, 0.15]
    # evaluator.plot_learning_curve(train_losses, val_losses)

    # # 使用例
    # # all_labels と all_preds は評価ループで得られたリストとします
    # # ここでは例としてランダムな値を生成していますが、実際の使用時は評価ループの結果を使用します
    # all_labels =  np.concatenate([np.random.randint(0, 2, size=2500), np.random.randint(3, 6, size=2500), np.random.randint(7, 15, size=5000)])
    # all_preds =  np.concatenate([np.random.randint(0, 4, size=2500), np.random.randint(5, 8, size=2500), np.random.randint(9, 12, size=2500), np.random.randint(13, 15, size=2500)])
    
    # np.random.shuffle(all_labels)
    # np.random.shuffle(all_preds)
    
    treatment_test_results = {
        'image_paths': [],
        'ground_truths': [],
        'anomaly_predictions': [],
        'treatment_predictions': []
    }
    
    with open('anomaly(treatment)_test_4class_results.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # ヘッダー行をスキップ
            
        for row in csvreader:
            treatment_test_results['image_paths'].append(row[0])
            treatment_test_results['ground_truths'].append(int(row[1]))
            treatment_test_results['anomaly_predictions'].append(int(row[2]))
            treatment_test_results['treatment_predictions'].append(int(row[3]))
    
    # 15クラスの詳細な混同行列
    evaluator.create_confusion_matrix(treatment_test_results['ground_truths'], treatment_test_results['treatment_predictions'], 15, mode='detailed')

    # 正常/異常の2クラスの混同行列
    evaluator.create_confusion_matrix(treatment_test_results['ground_truths'], treatment_test_results['treatment_predictions'], 15, mode='binary')
    
    # timeline_image.save(f'{output_dir}/timeline.jpg')
    evaluator.visualize_timeline('anomaly(treatment)_test_4class/treatment_test_results.csv')
    print('Timeline image saved as timeline.jpg')

if __name__ == '__main__':
    main()


