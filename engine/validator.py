import torch
from tqdm import tqdm

class Validator:
    """バリデーションループを実行するクラス。マルチラベル・シングルラベル両方に対応。"""

    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.is_multilabel = isinstance(criterion, torch.nn.BCEWithLogitsLoss)

    def validate(self, val_loader):
        """
        バリデーションを実行する。

        Args:
            val_loader: バリデーションデータのDataLoader。

        Returns:
            float: バリデーション全体の平均損失。
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, _, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                if self.is_multilabel:
                    # モデルが Unclear Head を持っているかチェック
                    if hasattr(self.model, 'has_unclear_head') and self.model.has_unclear_head:
                        main_classes = self.model.main_classes
                        
                        main_outputs = outputs[:, :main_classes]
                        unclear_outputs = outputs[:, main_classes:]
                        main_labels = labels[:, :main_classes]
                        unclear_labels = labels[:, main_classes:]
                        
                        main_loss = self.criterion(main_outputs, main_labels)
                        unclear_loss = self.criterion(unclear_outputs, unclear_labels)
                        loss = main_loss + unclear_loss
                    else:  # 6クラスのみの場合
                        loss = self.criterion(outputs, labels)
                else:  # シングルラベルの場合
                    loss = self.criterion(outputs, labels.long())
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)