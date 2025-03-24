import torch
import torch.nn as nn
import torchvision.models as models


class MultiLabelDetectionModel(nn.Module):
    def __init__(self, num_classes, pretrained=False, freeze_backbone=False):
        super(MultiLabelDetectionModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Initialize ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Get the number of features from ResNet
        feature_dim = self.resnet.fc.in_features
        
        # Remove the final fully connected layer
        self.resnet.fc = nn.Identity()
        
        # Optionally freeze backbone parameters
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Define the output layer
        self.output_layer = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
        x: Input tensor of shape (batch_size, n_image, c, h, w)

        Returns:
        Output tensor of shape (batch_size, num_classes)
        """
        features = self.resnet(x)  # (batch_size, num_features)
        out = self.output_layer(features)  # (batch_size, num_classes)

        return out


class MultiTaskDetectionModel(nn.Module):
    def __init__(self, num_classes=6, pretrained=False, freeze_backbone=False):
        """
        Args:
            num_classes: 全クラス数（例: 6, 7, 15）
            pretrained: ImageNetの重みを使用するかどうか
            freeze_backbone: バックボーンのパラメータを固定するかどうか
        """
        super(MultiTaskDetectionModel, self).__init__()
        
        # ResNet18の初期化（pretrainedの場合はImageNetの重みを利用）
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        feature_dim = self.resnet.fc.in_features
        
        # 最終層をIdentityに置き換えることで、特徴抽出部分のみを使用
        self.resnet.fc = nn.Identity()
        
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # クラス数に応じて出力層を設定
        if num_classes == 6:
            # 6クラスの場合は主クラスのみ
            self.main_head = nn.Linear(feature_dim, 6)
            self.unclear_head = None
        elif num_classes == 7:
            # 7クラスの場合は主クラス6 + 不鮮明クラス1
            self.main_head = nn.Linear(feature_dim, 6)
            self.unclear_head = nn.Linear(feature_dim, 1)
        elif num_classes == 15:
            # 15クラスの場合は主クラス6 + 不鮮明クラス9
            self.main_head = nn.Linear(feature_dim, 6)
            self.unclear_head = nn.Linear(feature_dim, 9)
        else:
            raise ValueError(f"Unsupported number of classes: {num_classes}")
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 入力テンソル (batch_size, c, h, w)
        Returns:
            連結された出力テンソル
            - 6クラスの場合: (batch_size, 6)
            - 7クラスの場合: (batch_size, 7)
            - 15クラスの場合: (batch_size, 15)
        """
        # バックボーンによる特徴抽出
        features = self.resnet(x)  # shape: (batch_size, feature_dim)
        
        # 主クラスの出力
        main_out = self.main_head(features)
        
        # クラス数に応じて出力を返す
        if self.num_classes == 6:
            return main_out
        else:
            # 不鮮明クラスの出力と連結
            unclear_out = self.unclear_head(features)
            return torch.cat([main_out, unclear_out], dim=1)

# ======================================
# モデルのインスタンス生成と出力例
# ======================================

def main():
    # 例: 主クラス6、かつ不鮮明クラス9 → 合計15クラスとして出力
    model = MultiTaskDetectionModel(num_classes=15, pretrained=True, freeze_backbone=False)

    # ダミー入力例 (batch_size=4, RGB画像 224x224)
    dummy_input = torch.randn(4, 3, 224, 224)

    # 順伝播
    output = model(dummy_input)  # 出力 shape: (4, 15)
    print("Concatenated output shape:", output.shape)

if __name__ == '__main__':
    main()


