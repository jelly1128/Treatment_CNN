from dataclasses import dataclass
from engine.inference import InferenceResult

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
        確率のしきい値を使用して、マルチソフトラベルをマルチライベルに変換します。
        
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
                hard_multilabel = [1 if probability > threshold else 0 for probability in probabilities]
                
                hard_multilabel_result.image_paths.append(image_path)
                hard_multilabel_result.multilabels.append(hard_multilabel)
                hard_multilabel_result.ground_truth_labels.append(labels)
            
            hard_multilabels_results[folder_name] = hard_multilabel_result
            
        return hard_multilabels_results
    