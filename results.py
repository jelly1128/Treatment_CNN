import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import csv

names = ['white light', 'lugol', 'indigo carmine', 'BLI・NBI', 'others']

plt.rcParams["font.family"] = 'DejaVu Serif'

def create_confusion_matrix(true_labels, pred_labels, confusion_matrix_img_file_name) :
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2, 3, 4])
    df = pd.DataFrame(cm, index=names, columns=names)
    sns.heatmap(df, cmap='Reds', annot = True, fmt="d",linewidth=.5)
    plt.xlabel("Prediction", fontsize=13)
    plt.ylabel("Ground truth", fontsize=13)
    plt.savefig(confusion_matrix_img_file_name)
    plt.close()

def create_csv(true_labels, pred_labels, csv_file_name):
    # システムごとの結果         
    with open(csv_file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['pred_labels', 'true_labels'])  # ヘッダー行を書き込む
        # リストの要素を行ごとに書き込む
        for index, (item1, item2) in enumerate(zip(pred_labels, true_labels)):
            writer.writerow([index + 1, item1, item2])
