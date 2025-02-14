from pathlib import Path
import logging
import torch
from torch.utils.data import DataLoader
import csv
from dataclasses import dataclass

# 推論結果を格納するクラス
@dataclass
class InferenceResult:
    """
    推論結果を格納するクラス。

    Attributes:
        probabilities: 予測確率のリスト
        labels: 正解ラベルのリスト
        image_paths: 画像パスのリスト
    """
    image_paths: list[str]
    probabilities: list[float]
    labels: list[int]

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
            folder_probabilities: 予測確率のリスト
            folder_labels: 正解ラベルのリスト
            folder_image_paths: 画像パスのリスト
        """
        results = InferenceResult([], [], [])
        
        with torch.no_grad():
            for images, image_paths, labels in test_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                sigmoid_outputs = torch.sigmoid(outputs[0])
                
                probabilities = sigmoid_outputs.cpu().numpy().round(4)
                batch_labels = labels.cpu().numpy().astype(int)
                
                results.image_paths.extend(image_paths)
                results.probabilities.extend(probabilities)
                results.labels.extend(batch_labels.tolist())
                
        return results

    def _save_results(self, save_dir: str, folder_name: str, results: InferenceResult):
        """
        推論結果を保存する。

        Args:
            save_dir: 保存先のディレクトリ
            folder_name: フォルダ名
            results: (probabilities, labels, image_paths)のタプル
        """
        save_path = Path(save_dir, folder_name)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 結果をCSVファイルに保存
        self._save_raw_results(save_path / f'raw_results_{folder_name}.csv', results)
        
        self._save_visualization(save_path / f'raw_results_{folder_name}.png', results)
        
        
        
    def _save_raw_results(self, csv_path: Path, results: InferenceResult):
        """
        推論結果をCSVファイルに保存する。
        """
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Image_Path'] + [f"Pred_Class_{i}" for i in range(len(results.probabilities[0]))] + \
                    [f"True_Class_{i}" for i in range(len(results.labels[0]))]
            writer.writerow(header)
            for img_path, probs, lbls in zip(results.image_paths, results.probabilities, results.labels):
                writer.writerow([img_path] + probs + lbls.tolist())
        
        logging.info(f"Saved raw results: {csv_path}")

    def run(self, save_dir, test_dataloaders):
        """
        推論を実行し、モデルの出力を返す。

        Args:
            save_dir: 結果保存ディレクトリ
            test_dataloaders: テスト用データローダーの辞書

        Returns:
            results: フォルダごとの推論結果を格納した辞書
        """
        results = {}
        
        for folder_name, test_dataloader in test_dataloaders.items():
            logging.info(f"Testing on {folder_name}...")
            
            # 推論実行
            folder_results = self._run_inference(test_dataloader)
            
            # 結果を保存
            self._save_results(save_dir, folder_name, folder_results)
            
            # 結果を辞書に格納
            results[folder_name] = folder_results

        logging.info("テスト完了。結果を保存しました。")

        return results