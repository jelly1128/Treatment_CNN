import csv
from engine.inference import InferenceResult
from labeling.label_converter import HardMultiLabelResult
from pathlib import Path
import numpy as np
from scipy.stats import entropy
import svgwrite
import re
from entropy_analyzer import EntropyDistributionAnalyzer, PredictionCertaintyAnalyzer
from sklearn.metrics import multilabel_confusion_matrix, classification_report


def main():
    
    # 解析器の初期化
    analyzer = PredictionCertaintyAnalyzer(save_dir_path=Path("debug_results/entropy_relation"), num_classes=7)
    
    # 全体のディレクトリパス
    dir_path = Path("/home/tanaka/Treatment_CNN/7class_resnet18_multitask_anomaly_test")
    
    # フォルダを走査
    for i in range(5):
        # 数字から始まる名前のフォルダを取得
        video_dirs = dir_path.glob(f"fold_{i}/*")
        
        for video_dir in video_dirs:
            # フォルダ名からフォルダ名を取得
            video_name = video_dir.name

            if re.match(r"^\d+", video_name):
                # print(video_name)
                raw_result_csv_path = dir_path / f'fold_{i}' / video_name / f'raw_results_{video_name}.csv'

                analyzer = PredictionCertaintyAnalyzer(save_dir_path=Path("debug_results/entropy_analyzer"), num_classes=6)
                inference_result = analyzer.load_inference_results(raw_result_csv_path)

                # 分析を実行
                analysis_result = analyzer.analyze_prediction_accuracy_correlation(inference_result)

                # 結果を可視化
                analyzer.visualize_accuracy_entropy_relation(analysis_result, inference_result, video_name)

                entropies_list = analyzer.analyze_prediction_certainty(inference_result)
                analyzer.visualize_certainty(entropies_list, video_name)

                entropy_analyzer = EntropyDistributionAnalyzer(save_dir_path=Path("debug_results/entropy_analyzer"))
                entropy_analyzed_result = entropy_analyzer.analyze_entropy_distribution(entropies_list, video_name)


if __name__ == "__main__":
    main()