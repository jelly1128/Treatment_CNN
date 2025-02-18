from dataclasses import dataclass
from engine.inference import InferenceResult
import csv
import logging
from pathlib import Path

@dataclass
class HardMultiLabelResult:
    """
    マルチラベルの結果をマルチラベルとシングルラベルに変換した結果を格納するクラス。

    Attributes:
        image_paths: 画像パスのリスト
        multilabels: マルチラベルの予測ラベルのリスト
        ground_truth_labels: マルチラベルの正解ラベルのリスト
    """
    image_paths: list[str]
    multilabels: list[list[int]]
    ground_truth_labels: list[list[int]]
    
@dataclass
class SingleLabelResult:
    """
    シングルラベルの結果を格納するクラス。
    
    Attributes:
        image_paths: 画像パスのリスト
        single_labels: シングルラベルの予測ラベルのリスト
        ground_truth_labels: シングルラベルの正解ラベルのリスト
    """
    image_paths: list[str]
    single_labels: list[int]
    ground_truth_labels: list[int]

class MultiToSingleLabelConverter:
    def __init__(self, inference_results_dict: dict[str, InferenceResult]):
        self.inference_results_dict = inference_results_dict
        
    def convert_soft_to_hard_multilabels(self, threshold: float = 0.5) -> dict[str, HardMultiLabelResult]:
        """
        確率のしきい値を使用して、マルチソフトラベルをマルチラベルに変換します。
        全てのラベルが閾値を超えていない場合、主ラベルの中で最も確率が高いものとその確率より高いサブラベルを選択します。
        
        Args:
            threshold (float): ラベルの割り当てを決定する確率のしきい値。
        
        Returns:
            dict[str, HardMultiLabelResult]: しきい値を超えるラベルのリストを含む辞書。
        """
        hard_multilabels_results = {}
        # フォルダごとにマルチラベルをマルチラベルに変換
        for folder_name, inference_result in self.inference_results_dict.items():
            hard_multilabel_result = HardMultiLabelResult(image_paths=[], multilabels=[], ground_truth_labels=[])
            
            for image_path, probabilities, labels in zip(inference_result.image_paths, inference_result.probabilities, inference_result.labels):
                # 通常の閾値処理
                hard_multilabel = [1 if prob > threshold else 0 for prob in probabilities]
                
                # 全てのラベルが0の場合の処理
                if sum(hard_multilabel) == 0:
                    # 主ラベル（0-5）の中で最も確率が高いものを見つける
                    main_probs = probabilities[:6]  # 主ラベルの確率
                    max_main_prob = max(main_probs)
                    max_main_idx = main_probs.index(max_main_prob)
                    
                    # 主ラベルの最大確率より高い確率を持つサブラベルを見つける
                    hard_multilabel = [0] * len(probabilities)
                    hard_multilabel[max_main_idx] = 1  # 最も確率の高い主ラベルを1に設定
                    
                    # サブラベル（6以降）で主ラベルの最大確率より高いものを1に設定
                    for i, prob in enumerate(probabilities[6:], start=6):
                        if prob > max_main_prob:
                            hard_multilabel[i] = 1
                
                hard_multilabel_result.image_paths.append(image_path)
                hard_multilabel_result.multilabels.append(hard_multilabel)
                hard_multilabel_result.ground_truth_labels.append(labels)
            
            hard_multilabels_results[folder_name] = hard_multilabel_result
            
        return hard_multilabels_results
    
    def save_hard_multilabel_results(self, hard_multilabel_results: dict[str, HardMultiLabelResult], save_dir: Path, methods: str = "threshold"):
        """
        マルチラベルの変換結果をCSVファイルに保存します。

        Args:
            hard_multilabel_results (dict[str, HardMultiLabelResult]): 変換結果の辞書
            save_dir (Path): 保存先ディレクトリ
            method (str): 変換手法の名前（ファイル名に使用）
        """
        # 保存先ディレクトリが存在しない場合は作成
        save_dir = save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # フォルダごとの結果を保存
        for folder_name, result in hard_multilabel_results.items():
            folder_dir = save_dir / folder_name / methods
            folder_dir.mkdir(parents=True, exist_ok=True)
            # フォルダごとのCSVファイルを作成
            folder_file = folder_dir / f'{methods}_results_{folder_name}.csv'
            
            with open(folder_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                # ヘッダー行
                header = ['Image_Path']
                header.extend([f'Label_{i}' for i in range(len(result.multilabels[0]))])  # 予測ラベル
                header.extend([f'True_Label_{i}' for i in range(len(result.ground_truth_labels[0]))])  # 正解ラベル
                writer.writerow(header)
                
                # データ行
                for img_path, pred_labels, true_labels in zip(
                    result.image_paths,
                    result.multilabels,
                    result.ground_truth_labels
                ):
                    row = [img_path]
                    row.extend(pred_labels)
                    row.extend(true_labels)
                    writer.writerow(row)
            
            logging.info(f'Saved multilabel results for {folder_name} to {folder_file}')