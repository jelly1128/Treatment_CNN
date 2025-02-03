import torch
import torch.nn as nn
import torchvision.models as models

    
class CNN(nn.Module):
    def __init__(self, n_class, pretrain=False):
        super(CNN, self).__init__()

        self.n_class = n_class
        
        # ResNet18
        if pretrain:
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
        self.output_layer = nn.Linear(num_features, n_class)

    def forward(self, input):
        # cnn features
        #print(input.size()) (batch_size, n_image, c, h, w)
        
        # FCに特徴量を入力
        out = self.output_layer(self.resnet(input))
        
        return out
    

class AnomalyDetectionModel(nn.Module):
    def __init__(self, n_class, pretrain=False, freeze_backbone=False):
        super(AnomalyDetectionModel, self).__init__()

        self.n_class = n_class
        
        # ResNet18
        if pretrain:
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
        self.output_layer = nn.Linear(num_features, n_class)

    def forward(self, input):
        # cnn features
        #print(input.size()) (batch_size, n_image, c, h, w)
        
        # FCに特徴量を入力
        out = self.output_layer(self.resnet(input))
        
        return out
    

class TreatmentClassificationModel(nn.Module):
    def __init__(self, n_class, n_image, pretrain=False, freeze_backbone=False, hidden_size=128, n_lstm=2):
        super(TreatmentClassificationModel, self).__init__()
        """
        処置分類モデルの初期化
        
        :param n_class: 
        :param n_image: 
        :param anomaly_detector: 
        """
        self.n_image = n_image
        
        # ResNet18
        if pretrain:
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
        self.output_layer = nn.Linear(hidden_size, n_class)

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
    def __init__(self, n_class, pretrain=False, freeze_backbone=False):
        super(MultiLabelDetectionModel, self).__init__()

        self.n_class = n_class
        
        # ResNet18
        if pretrain:
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
        self.output_layer = nn.Linear(num_features, n_class)

    def forward(self, input):
        # cnn features
        #print(input.size()) (batch_size, n_image, c, h, w)
        
        # FCに特徴量を入力
        out = self.output_layer(self.resnet(input))
        
        # Sigmoid関数を適用して各クラスの確率を得る（BCEWithLogitsLossを使用する場合，いらない）
        # out = torch.sigmoid(out)
        
        return out