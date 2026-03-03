import torch
from torch.utils.data import DataLoader
from evaluate.result_types import InferenceResult, SingleLabelResult

class Inference:
    def __init__(self, model: torch.nn.Module, device: str):
        """
        推論エンジンの初期化。

        Args:
            model: 推論に使用するモデル
            device: 使用するデバイス（'cuda' または 'cpu'）
        """
        self.model = model
        self.device = device
        self.model.eval()  # モデルを評価モードに設定

    def _run_inference(self, test_dataloader: DataLoader) -> InferenceResult:
        """
        1つのデータローダーに対して推論を実行する。

        Args:
            test_dataloader: テスト用データローダー

        Returns:
            InferenceResult: 推論結果
            - image_paths: 画像パスのリスト
            - probabilities: 予測確率のリスト
            - labels: 正解ラベルのリスト
        """
        results = InferenceResult(image_paths=[], probabilities=[], labels=[])
        
        with torch.no_grad():
            for images, image_paths, labels in test_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                sigmoid_outputs = torch.sigmoid(outputs)  # バッチ全体 (B, C)

                batch_probs = [[round(float(p), 4) for p in row] for row in sigmoid_outputs.tolist()]
                labels = labels.cpu().numpy().astype(int).tolist()

                results.image_paths.extend(image_paths)
                results.probabilities.extend(batch_probs)
                results.labels.extend(labels)
        
        return results

    def _run_inference_single_label(self, test_dataloader: DataLoader) -> SingleLabelResult:
        """
        シングルラベル用の推論を実行する。

        Args:
            test_dataloader: テスト用データローダー

        Returns:
            SingleLabelResult: 画像パス、予測ラベル（整数）、正解ラベル（整数）
        """
        results = SingleLabelResult(image_paths=[], single_labels=[], ground_truth_labels=[])
        with torch.no_grad():
            for images, image_paths, labels in test_dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(dim=1).cpu().tolist()
                labels = labels.cpu().tolist()
                results.image_paths.extend(image_paths)
                results.single_labels.extend(preds)
                results.ground_truth_labels.extend(labels)
        return results

    def run(self, test_dataloaders: dict[str, DataLoader], mode: str = 'multi_label') -> dict[str, object]:
        """
        推論を実行し、モデルの出力を返す。

        Args:
            test_dataloaders: テスト用データローダーの辞書
            mode: 'single_label'ならSingleLabelResult, それ以外はInferenceResult

        Returns:
            results: フォルダごとの推論結果を格納した辞書
            - video_name: フォルダ名
            - results: 推論結果
        """
        results = {}

        for video_name, test_dataloader in test_dataloaders.items():
            if mode == 'single_label':
                folder_results = self._run_inference_single_label(test_dataloader)
            else:
                folder_results = self._run_inference(test_dataloader)
            results[video_name] = folder_results

        return results