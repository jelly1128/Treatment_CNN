import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ModelEvaluator:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_learning_curve(self, loss_history: dict):
        epochs = range(1, len(loss_history['train']) + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, loss_history['train'], label='Train Loss')
        plt.plot(epochs, loss_history['val'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Learning Curve')
        plot_path = os.path.join(self.save_dir, 'learning_curve.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Learning curve saved to {plot_path}")
    
    def save_loss_to_csv(self, loss_history: dict):
        csv_path = os.path.join(self.save_dir, 'loss_history.csv')
        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Epoch", "Train Loss", "Validation Loss"])
            for epoch, (train_loss, val_loss) in enumerate(zip(loss_history['train'], loss_history['val']), start=1):
                writer.writerow([epoch, train_loss, val_loss])
        print(f"Loss history saved to {csv_path}")
        
    def plot_roc_curve(self, true_labels, predicted_probabilities, class_index, save_directory):
        """
        Plots the ROC curve and saves it to the specified directory.
        """
        false_positive_rates, true_positive_rates, _ = roc_curve(
            true_labels,
            predicted_probabilities
        )
        area_under_curve = auc(false_positive_rates, true_positive_rates)
        youden_index = true_positive_rates - false_positive_rates
        best_threshold_index = np.argmax(youden_index)
        best_false_positive_rate = false_positive_rates[best_threshold_index]
        best_true_positive_rate = true_positive_rates[best_threshold_index]
        best_threshold = _[best_threshold_index]

        plt.plot(false_positive_rates, true_positive_rates, color='darkorange',
                 lw=2, label=f'ROC curve (AUC = {area_under_curve:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                 label='Random Classifier')
        plt.scatter(best_false_positive_rate, best_true_positive_rate,
                    color='red', marker='o',
                    label=f'Youden\'s Index (Best Threshold = {best_threshold:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Class {class_index}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_directory,
                                 f'roc_curve_class_{class_index}.png'))
        plt.close()

    def evaluate_classification(y_true, y_pred):
        """
        分類メトリクスを計算する。
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return accuracy, precision, recall, f1