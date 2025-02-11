import os
import logging
import torch


class Inference:
    def __init__(self, model, device):
        """
        推論エンジンの初期化。

        Args:
            model: 推論に使用するモデル
            device: 使用するデバイス（'cuda' または 'cpu'）
        """
        self.model = model
        self.device = device
        self.model.eval()  # モデルを評価モードに設定

    def run(self, save_dir, test_dataloaders):
        """
        推論を実行し、モデルの出力を返す。

        Args:
            dataloader: データローダー

        Returns:
            outputs: モデルの生の出力（ロジットや確率）
            labels: 正解ラベル
            image_paths: 画像のパス
        """
        
        results = {}
        
        for folder_name, test_dataloader in test_dataloaders.items():
            logging.info(f"Testing on {folder_name}...")
            # 結果保存用フォルダを作成
            save_path = os.path.join(save_dir, folder_name)
            os.makedirs(save_path, exist_ok=True)
            
            folder_probabilities = []
            folder_labels = []
            folder_image_paths = []
            
            with torch.no_grad():
                for images, image_paths, labels in test_dataloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(images)
                    # outputsの各要素に対してsigmoidを適用し、それを各クラスごとに出力する
                    sigmoid_outputs = torch.sigmoid(outputs[0])

                    # 出力が多次元の場合、ネストされたリストに対してフォーマットを適用する
                    # formatted_outputs = [f"{prob:.4f}" for prob in sigmoid_outputs.tolist()]
                    formatted_outputs = [round(prob, 4) for prob in sigmoid_outputs.tolist()]
                    
                    # print(image_paths[0], labels[0].cpu().numpy(), formatted_outputs)
                    
                    folder_probabilities.append(formatted_outputs)
                    folder_labels.append(labels[0].cpu().numpy().astype(int))
                    folder_image_paths.extend(image_paths)
                    
            # フォルダごとの結果を辞書に保存
            results[folder_name] = (folder_probabilities, folder_labels, folder_image_paths)

        logging.info("テスト完了。結果を保存しました。")

        return results