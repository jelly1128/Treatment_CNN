import csv
from engine.inference import InferenceResult
from labeling.label_converter import HardMultiLabelResult
from pathlib import Path
import numpy as np
from scipy.stats import entropy
import svgwrite
import re
from entropy_analyzer import EntropyDistributionAnalyzer
from sklearn.metrics import multilabel_confusion_matrix, classification_report

class PredictionCertaintyAnalyzer:
    def __init__(self, save_dir_path: Path, num_classes: int):
        """
        推論結果を解析するクラス。
        """
        self.save_dir_path = save_dir_path
        self.save_dir_path.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        self.num_scene_classes = 6

    def load_inference_results(self, csv_path: Path) -> InferenceResult:
        """
        CSVファイルから推論結果を読み込む。

        Args:
            csv_path: 読み込むCSVファイルのパス

        Returns:
            InferenceResult: 読み込んだ推論結果
        """
        try:
            image_paths, probabilities, labels = self._read_inference_results_csv(csv_path)
            return InferenceResult(image_paths=image_paths, probabilities=probabilities, labels=labels)
        except Exception as e:
            raise

    def _read_inference_results_csv(self, csv_path: Path):
        """CSVファイルを読み込み、データを分割するヘルパーメソッド
        
        Args:
            csv_path (Path): 読み込むCSVファイルのパス
            
        Returns:
            tuple:
                list[str]: 画像パスのリスト
                list[list[float]]: 確率値のリスト
                list[list[int]]: ラベルのリスト
        """
        image_paths = []
        probabilities = []
        ground_truth_labels = []

        with open(csv_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)  # ヘッダーをスキップ

            for row in reader:
                image_paths.append(row[0])
                # 確率値とラベルの区切り位置を計算
                num_probabilities = (len(row) - 1) // 2
                
                # 確率値とラベルを分割して追加
                probabilities.append(
                    list(map(float, row[1:num_probabilities + 1]))
                )
                ground_truth_labels.append(
                    list(map(int, row[num_probabilities + 1:]))
                )

        return image_paths, probabilities, ground_truth_labels

    def load_threshold_results(self, csv_path: Path) -> HardMultiLabelResult:
        """
        CSVファイルから閾値を適用した結果を読み込む。
        """
        try:
            image_paths, multi_labels, ground_truth_labels = self._read_threshold_results_csv(csv_path)
            return HardMultiLabelResult(image_paths=image_paths, multi_labels=multi_labels, ground_truth_labels=ground_truth_labels)
        except Exception as e:
            raise

    def _read_threshold_results_csv(self, csv_path: Path):
        """CSVファイルを読み込み、データを分割するヘルパーメソッド
        
        Args:
            csv_path (Path): 読み込むCSVファイルのパス
            
        Returns:
            tuple:
                list[str]: 画像パスのリスト
                list[list[float]]: 確率値のリスト
                list[list[int]]: ラベルのリスト
        """
        image_paths = []
        multi_labels = []
        ground_truth_labels = []

        with open(csv_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)  # ヘッダーをスキップ

            for row in reader:
                image_paths.append(row[0])
                # 確率値とラベルの区切り位置を計算
                num_multi_labels = (len(row) - 1) // 2
                
                # 確率値とラベルを分割して追加
                multi_labels.append(
                    list(map(float, row[1:num_multi_labels + 1]))
                )
                ground_truth_labels.append(
                    list(map(int, row[num_multi_labels + 1:]))
                )

        return image_paths, multi_labels, ground_truth_labels
    
    
    def analyze_prediction_certainty(self, inference_result: InferenceResult) -> list[float]:
        """
        予測確信度を解析する。
        """
        # 予測確信度を計算
        normalized_entropies = self._calculate_prediction_certainty(inference_result.probabilities)

        return normalized_entropies

    
    def _calculate_prediction_certainty(self, probabilities: list[list[float]]) -> list[float]:
        """
        予測確信度を計算するヘルパーメソッド
        
        Args:
            inference_result (InferenceResult): 推論結果
            
        Returns:
            float: 予測確信度
        """
        normalized_entropies = []
        # 推論結果（確率）の正規化エントロピーを計算
        for i in range(len(probabilities)):
            # 正規化エントロピー
            # すべてのクラスを使用して算出する場合
            # normalized_entropy = entropy(probabilities[i], base=2) / np.log2(self.num_classes)
            # シーンクラスのみを使用して算出する場合
            normalized_entropy = entropy([probabilities[i][j] for j in range(self.num_scene_classes)], base=2) / np.log2(self.num_scene_classes)
            normalized_entropies.append(normalized_entropy)

            # debug用
            # if i % 100 == 0:
            #     print(probabilities[i])
            #     print(normalized_entropy)

        return normalized_entropies
    

    def visualize_certainty(self, entropies_list: list[float], video_name: str):
        """
        予測確信度を可視化する。
        
        Args:
            entropies_list (list[float]): 予測確信度のリスト
            video_name (str): 動画名
        """
        n_images = len(entropies_list)
        
        # 時系列の画像を作成
        timeline_width = n_images
        timeline_height = n_images // 10
        
        # 保存パスの設定
        save_file = self.save_dir_path / f'{video_name}certainty_timeline.svg'
            
        # SVGドキュメントの作成
        dwg = svgwrite.Drawing(str(save_file), size=(timeline_width, timeline_height))
        dwg.add(dwg.rect((0, 0), (timeline_width, timeline_height), fill='white'))

        for i in range(n_images):
            x1 = i * (timeline_width // n_images)
            x2 = (i + 1) * (timeline_width // n_images)
            
            colored_certainty = int(255 * entropies_list[i])
            color = (colored_certainty, colored_certainty, colored_certainty)
            color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
            dwg.add(dwg.rect((x1, 0), (x2-x1, timeline_height), fill=color_hex))

        # SVGファイルを保存
        dwg.save()


def main():
    # 一つずつ
    raw_result_csv_path = Path("/home/tanaka/Treatment_CNN/7class_resnet18_multitask_anomaly_test/fold_0/20210524100043_000001-001/raw_results_20210524100043_000001-001.csv")
    threshold_result_csv_path = Path("/home/tanaka/Treatment_CNN/7class_resnet18_multitask_anomaly_test/fold_0/20210524100043_000001-001/threshold_50%/threshold_50%_results_20210524100043_000001-001.csv")
    
    # raw_result_csv_path = Path("/home/tanaka/Treatment_CNN/6class_resnet18_multitask_anomaly_test/fold_0/20210531112330_000001-001/raw_results_20210531112330_000001-001.csv")
    # threshold_result_csv_path = Path("/home/tanaka/Treatment_CNN/6class_resnet18_multitask_anomaly_test/fold_0/20210531112330_000001-001/threshold_50%/threshold_50%_results_20210531112330_000001-001.csv")

    analyzer = PredictionCertaintyAnalyzer(save_dir_path=Path("debug_results"), num_classes=7)
    raw_result = analyzer.load_inference_results(raw_result_csv_path)
    threshold_result = analyzer.load_threshold_results(threshold_result_csv_path)

    # 計算
    entropies_list = analyzer.analyze_prediction_certainty(raw_result)

    # 可視化
    # analyzer.visualize_certainty(entropies_list, '20210524100043_000001-001')

    entropies_result = {
        'image_paths': raw_result.image_paths,
        'predictions': threshold_result.multi_labels,
        'ground_truths': threshold_result.ground_truth_labels,
        'entropies': entropies_list
    }

    confusion_matrix = multilabel_confusion_matrix(threshold_result.ground_truth_labels, threshold_result.multi_labels, labels=list(range(7)))
    print(confusion_matrix)
    print(classification_report(threshold_result.ground_truth_labels, threshold_result.multi_labels, labels=list(range(7))))

    entropied_threshold_results = {
        'predictions': [],
        'ground_truths': []
    }

    for i in range(len(entropies_list)):
        if entropies_list[i] < 0.3:
            entropied_threshold_results['predictions'].append(threshold_result.multi_labels[i])
            entropied_threshold_results['ground_truths'].append(threshold_result.ground_truth_labels[i])

    print(len(entropied_threshold_results['predictions']))
    entropied_threshold_cm = multilabel_confusion_matrix(entropied_threshold_results['ground_truths'], entropied_threshold_results['predictions'], labels=list(range(7)))
    print(entropied_threshold_cm)
    print(classification_report(entropied_threshold_results['ground_truths'], entropied_threshold_results['predictions'], labels=list(range(7))))

    # # 全体
    # dir_path = Path("/home/tanaka/Treatment_CNN/7class_resnet18_multitask_anomaly_test")
    
    # for i in range(5):
    #     # 数字から始まる名前のフォルダを取得
    #     video_dirs = dir_path.glob(f"fold_{i}/*")
        
    #     for video_dir in video_dirs:
    #         # フォルダ名からフォルダ名を取得
    #         video_name = video_dir.name

    #         if re.match(r"^\d+", video_name):
    #             # print(video_name)
    #             raw_result_csv_path = dir_path / f'fold_{i}' / video_name / f'raw_results_{video_name}.csv'

    #             analyzer = PredictionCertaintyAnalyzer(save_dir_path=Path("debug_results"), num_classes=6)
    #             raw_result = analyzer.load_inference_results(raw_result_csv_path)

    #             entropies_list = analyzer.analyze_prediction_certainty(raw_result)

    #             # analyzer.visualize_certainty(entropies_list, video_name)

    #             entropy_analyzer = EntropyDistributionAnalyzer(save_dir_path=Path("debug_results/7class"))
    #             entropy_analyzed_result = entropy_analyzer.analyze_entropy_distribution(entropies_list, video_name)


if __name__ == "__main__":
    main()