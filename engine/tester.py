import os
import torch
import numpy as np
import pandas as pd
import logging
import csv
from torch.utils.data import DataLoader
from data.visualization import visualize_multilabel_timeline, visualize_ground_truth_timeline
from model.setup_models import setup_model
from data.dataloader import create_multilabel_test_dataloaders
from engine.inference import Inference
# from evaluate.metrics import ClassificationMetricsCalculator
# from utils.file_utils import save_results_to_csv

class Tester:
    def __init__(self, config, device, num_gpus, test_dirs):
        self.config = config
        self.device = device
        self.num_gpus = num_gpus
        self.test_dataloaders = create_multilabel_test_dataloaders(config, test_dirs, num_gpus)
        self.model = setup_model(config, device, num_gpus, mode='test')
        # self.metrics_calculator = ClassificationMetricsCalculator()
        self.inference = Inference(self.model, device)
        # self.output_analyzer = OutputAnalyzer(threshold=0.5)

    def test(self):
        # 推論
        results = self.inference.run(self.config.paths.save_dir, self.test_dataloaders)
        
        for result in range(results):
            print(len(results[result]))
        print(len(results))
        # print(results)
        
        # os._exit(0)
        # 出力を解析
        # metrics, predictions = self.output_analyzer.analyze(outputs, labels)

        # print("Test Metrics:")
        # print(f"Accuracy: {metrics['accuracy']:.4f}")
        # print(f"Precision: {metrics['precision']:.4f}")
        # print(f"Recall: {metrics['recall']:.4f}")
        # print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        
    
        # """
        # テストを実行する。
        # """
        # all_probabilities = []
        # all_labels = []
        # all_image_paths = []

        # with torch.no_grad():
        #     for images, image_paths, labels in test_dataloader:
        #         images = images.to(self.device)
        #         outputs = self.model(images)
        #         probabilities = torch.sigmoid(outputs[0]).cpu().numpy()
        #         all_probabilities.append(probabilities)
        #         all_labels.append(labels[0].cpu().numpy())
        #         all_image_paths.extend(image_paths)

        # # 結果を保存
        # self._save_results(all_probabilities, all_labels, all_image_paths, folder_name)

        # # メトリクスを計算
        # metrics = self.metrics_calculator.calculate(all_labels, all_preds)
        # return metrics

    def _save_results(self, probabilities, labels, image_paths, folder_name):
        """
        結果をCSVファイルに保存する。
        """
        save_dir = os.path.join(self.config.paths.save_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        # CSVファイルに保存
        output_csv_file = os.path.join(save_dir, 'multilabels_test_with_labels.csv')
        save_results_to_csv(output_csv_file, image_paths, probabilities, labels)

        # タイムラインの可視化
        df = pd.DataFrame(
            data=np.hstack([(np.array(probabilities) >= 0.5).astype(int), np.array(labels)]),
            columns=[f"Predicted_Class_{i}" for i in range(self.config.test.num_classes)] +
                    [f"Label_Class_{i}" for i in range(self.config.test.num_classes)]
        )
        df['Image_Path'] = image_paths

        visualize_multilabel_timeline(df, save_dir, "predicted", self.config.test.num_classes)
        visualize_ground_truth_timeline(df, save_dir, "ground_truth")