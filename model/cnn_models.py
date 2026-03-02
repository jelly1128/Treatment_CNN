import torch
import torch.nn as nn
import torchvision.models as models


from config.schema import ModelArchitecture

class SingleLabelClassificationModel(nn.Module):
    """シングルラベル分類用のモデル。"""
    def __init__(
        self, 
        architecture: ModelArchitecture | str,
        num_classes: int,
        pretrained: bool, 
        freeze_backbone: bool
    ):
        super(SingleLabelClassificationModel, self).__init__()
        self.num_classes = num_classes

        # Enum を文字列に変換（Enum でも str でも受け取れるように）
        if isinstance(architecture, ModelArchitecture):
            arch_str = architecture.value
        else:
            arch_str = architecture
        
        # アーキテクチャごとに初期化
        if arch_str == 'resnet18':
            self.network = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.fc.in_features
            self.network.fc = nn.Identity()
        elif arch_str == 'resnet34':
            self.network = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.fc.in_features
            self.network.fc = nn.Identity()
        elif arch_str == 'resnet50':
            self.network = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.fc.in_features
            self.network.fc = nn.Identity()
        elif arch_str == 'resnet101':
            self.network = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.fc.in_features
            self.network.fc = nn.Identity()
        elif arch_str == 'resnet152':
            self.network = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.fc.in_features
            self.network.fc = nn.Identity()
        elif arch_str == 'efficientnet-b0':
            self.network = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.classifier[1].in_features
            self.network.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model architecture: {arch_str}")
        
        # backbone部分のパラメータ固定化を選択
        if freeze_backbone:
            for param in self.network.parameters():
                param.requires_grad = False
        self.output_layer = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.network(x)
        out = self.output_layer(features)
        return out


class MultiTaskClassificationModel(nn.Module):
    """
    マルチタスク分類用のモデル。クラス数に応じて出力層を動的に構築。
    
    主クラス：シーン6クラス
    不鮮明クラス：0クラス（6クラスの場合） / 1クラス（7クラスの場合） / 9クラス（15クラスの場合）

    num_classes = 主クラス + 不鮮明クラス
    """
    MAIN_CLASSES = 6  # 主クラスは常に6クラス

    def __init__(
        self, 
        architecture: ModelArchitecture | str,
        num_classes: int = 6,
        pretrained: bool = False,
        freeze_backbone: bool = False
    ):
        super(MultiTaskClassificationModel, self).__init__()

        # バリデーション
        if num_classes < self.MAIN_CLASSES:
            raise ValueError(
                f"num_classes は {self.MAIN_CLASSES} 以上が必要です（指定値: {num_classes}）"
            )
        
        # Enum を文字列に変換
        if isinstance(architecture, ModelArchitecture):
            arch_str = architecture.value
        else:
            arch_str = architecture
        
        # バックボーン構築
        feature_dim = self._build_backbone(arch_str, pretrained, freeze_backbone)
        
        # ヘッド構築
        self.num_classes = num_classes
        self.main_head = nn.Linear(feature_dim, self.MAIN_CLASSES)
        
        unclear_classes = num_classes - self.MAIN_CLASSES
        if unclear_classes > 0:
            self.unclear_head = nn.Linear(feature_dim, unclear_classes)
        else:
            self.unclear_head = None
    
    def _build_backbone(self, arch_str: str, pretrained: bool, freeze_backbone: bool) -> int:
        """バックボーンを構築して feature_dim を返す"""
        if arch_str == 'resnet18':
            self.network = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.fc.in_features
            self.network.fc = nn.Identity()
        elif arch_str == 'resnet34':
            self.network = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.fc.in_features
            self.network.fc = nn.Identity()
        elif arch_str == 'resnet50':
            self.network = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.fc.in_features
            self.network.fc = nn.Identity()
        elif arch_str == 'resnet101':
            self.network = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.fc.in_features
            self.network.fc = nn.Identity()
        elif arch_str == 'resnet152':
            self.network = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.fc.in_features
            self.network.fc = nn.Identity()
        elif arch_str == 'efficientnet-b0':
            self.network = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = self.network.classifier[1].in_features
            self.network.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model architecture: {arch_str}")
        
        if freeze_backbone:
            for param in self.network.parameters():
                param.requires_grad = False
        
        return feature_dim
    
    @property
    def main_classes(self):
        return self.MAIN_CLASSES
    
    @property
    def unclear_classes(self):
        return self.num_classes - self.MAIN_CLASSES

    @property
    def has_unclear_head(self) -> bool:
        """不鮮明ヘッドを持つかどうか"""
        return self.unclear_head is not None


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
        features = self.network(x)  # shape: (batch_size, feature_dim)
        
        # 主クラスの出力
        main_out = self.main_head(features)
        
        # クラス数に応じて出力を返す
        if self.unclear_head is None:
            return main_out
        
        unclear_out = self.unclear_head(features)
        return torch.cat([main_out, unclear_out], dim=1)

# ======================================
# モデルのインスタンス生成と出力例
# ======================================

def main():
    # 例: 主クラス6、かつ不鮮明クラス9 → 合計15クラスとして出力
    model = MultiTaskClassificationModel(num_classes=15, pretrained=True, freeze_backbone=False)

    # ダミー入力例 (batch_size=4, RGB画像 224x224)
    dummy_input = torch.randn(4, 3, 224, 224)

    # 順伝播
    output = model(dummy_input)  # 出力 shape: (4, 15)
    print("Concatenated output shape:", output.shape)

if __name__ == '__main__':
    main()


