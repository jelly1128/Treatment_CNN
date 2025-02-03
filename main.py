import torch
from pathlib import Path
from config.config_loader import load_config
from data.dataloader import create_dataloaders
from model.model import MultiLabelDetectionModel
from model.loss import get_criterion
from engine.trainer import Trainer
from engine.validator import Validator
from utils.logger import setup_logging
from utils.visualization import plot_learning_curve

def main():
    # 設定読み込み
    config = load_config(Path("config.yaml"))
    setup_logging(Path(config.paths.save_dir))
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # データローダー
    train_loader, val_loader = create_dataloaders(config, fold=config.training.fold)
    
    # モデルと損失関数
    model = MultiLabelDetectionModel(
        num_classes=config.training.num_classes,
        pretrained=config.training.pretrained,
        freeze_backbone=config.training.freeze_backbone,
    ).to(device)
    criterion = get_criterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    
    # 学習・検証エンジン
    trainer = Trainer(model, optimizer, criterion, device)
    validator = Validator(model, criterion, device)
    
    # 学習ループ
    train_losses, val_losses = [], []
    for epoch in range(config.training.max_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = validator.validate(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # ログ出力
        logging.info(f"Epoch {epoch+1}/{config.training.max_epochs} | "
                     f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # モデル保存
        if val_loss == min(val_losses):
            torch.save(model.state_dict(), Path(config.paths.save_dir) / "best_model.pth")
    
    # 学習曲線の可視化
    plot_learning_curve(train_losses, val_losses, Path(config.paths.save_dir) / "learning_curve.png")

if __name__ == "__main__":
    main()