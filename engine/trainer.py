from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        for images, _, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        
        return total_loss / len(train_loader)