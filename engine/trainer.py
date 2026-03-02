from tqdm import tqdm
import torch.nn as nn

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.is_multilabel = isinstance(criterion, nn.BCEWithLogitsLoss)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        for images, _, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)

            if self.is_multilabel:  # マルチラベルの場合
                if hasattr(self.model, 'has_unclear_head') and self.model.has_unclear_head: # 不鮮明クラスのヘッドがある場合
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
            loss.backward()
            self.optimizer.step()
        
        return total_loss / len(train_loader)