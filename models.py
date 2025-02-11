import torch
import torch.nn as nn
import torchvision.models as models

    
class CNN(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(CNN, self).__init__()

        self.num_classes = num_classes
        
        # ResNet18
        if pretrained:
           self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
           for param in self.resnet.parameters():
               param.requires_grad = False
        else:
           self.resnet = models.resnet18()
        
        # CNNから出力される特徴量のサイズ(512)
        num_features = self.resnet.fc.in_features
        
        # 最後の層取り除く
        self.resnet.fc = nn.Identity()
        
        #FC
        self.output_layer = nn.Linear(num_features, num_classes)

    def forward(self, input):
        # cnn features
        #print(input.size()) (batch_size, n_image, c, h, w)
        
        # FCに特徴量を入力
        out = self.output_layer(self.resnet(input))
        
        return out
    

class AnomalyDetectionModel(nn.Module):
    def __init__(self, num_classes, pretrained=False, freeze_backbone=False):
        super(AnomalyDetectionModel, self).__init__()

        self.num_classes = num_classes
        
        # ResNet18
        if pretrained:
           self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
           self.resnet = models.resnet18()
        
        # CNNから出力される特徴量のサイズ(512)
        num_features = self.resnet.fc.in_features
        
        # 最後の層取り除く
        self.resnet.fc = nn.Identity()
        
        # パラメータ固定
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        #FC
        self.output_layer = nn.Linear(num_features, num_classes)

    def forward(self, input):
        # cnn features
        #print(input.size()) (batch_size, n_image, c, h, w)
        
        # FCに特徴量を入力
        out = self.output_layer(self.resnet(input))
        
        return out
    

class TreatmentClassificationModel(nn.Module):
    def __init__(self, num_classes, n_image, pretrained=False, freeze_backbone=False, hidden_size=128, n_lstm=2):
        super(TreatmentClassificationModel, self).__init__()
        """
        処置分類モデルの初期化
        
        :param num_classes: 
        :param n_image: 
        :param anomaly_detector: 
        """
        self.n_image = n_image
        
        # ResNet18
        if pretrained:
           self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
           self.resnet = models.resnet18()
        
        # CNNから出力される特徴量のサイズ(512)
        num_features = self.resnet.fc.in_features
        
        # 最後の層取り除く
        self.resnet.fc = nn.Identity()
        
        # パラメータ固定
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # LSTM
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=n_lstm, batch_first=True)

        #FC
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        # cnn features
        #print(input.size()) (batch_size, n_image, c, h, w)
        cnn_features =[]
        for i in range(self.n_image):
            cnn_feature = self.resnet(input[:,i])    # (batch_size, num_features(512))
            cnn_feature = cnn_feature.unsqueeze(1)       # (batch_size, 1, num_features(512))
            cnn_features.append(cnn_feature)
        cnn_features = torch.cat(cnn_features, dim=1)    # (batch_size, n_image, num_features(512))
        
        # LSTMに特徴量を入力
        lstm_outputs, (ht, ct) = self.lstm(cnn_features)
        
        # 最後の隠れ層の出力(最後のフレームの予測)を全結合層に入力し、予測値を出力
        fc_out = self.output_layer(lstm_outputs[:, -1, :])
        
        return fc_out


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
        self.output_layer = nn.Linear(feature_dim, 6) # 主クラス用の出力層
        if num_classes == 7:
            self.sub_output_layer = nn.Linear(feature_dim, 1) # 副クラス用の出力層
        elif num_classes == 15:
            self.sub_output_layer = nn.Linear(feature_dim, 9) # 副クラス用の出力層

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

        if self.num_classes == 7:
            sub_out = self.sub_output_layer(features)  # (batch_size, 1)
            out = torch.cat((out, sub_out), dim=1)
            
        elif self.num_classes == 15:
            sub_out = self.sub_output_layer(features)  # (batch_size, 9)
            out = torch.cat((out, sub_out), dim=1)

        return out
