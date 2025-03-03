import torch
from tqdm import tqdm

class Validator:
    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, _, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                # モデルの出力次元数を確認
                if outputs.shape[1] > 6:  # マルチタスクモデルの場合
                    main_outputs, unclear_outputs = outputs[:, :6], outputs[:, 6:]
                    main_labels, unclear_labels = labels[:, :6], labels[:, 6:] 
                    main_loss = self.criterion(main_outputs, main_labels)
                    unclear_loss = self.criterion(unclear_outputs, unclear_labels)
                    loss = main_loss + unclear_loss

                else:  # 6クラスのみの場合
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)