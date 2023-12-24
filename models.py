import torch
import torch.nn as nn
import torchvision.models as models

from resnet3d import r3d_18, r3d_34, r2plus1d_18, r2plus1d_34


class CNN_LSTM(nn.Module):
    def __init__(self, n_class, hidden_size, n_cnn, n_rnn):
        super(CNN_LSTM, self).__init__()

        self.n_class = n_class
        self.hidden_size = hidden_size
        self.n_cnn = n_cnn
        self.n_rnn = n_rnn

        #(1) CNN
        # ResNet18
        self.resnets = nn.ModuleList([models.resnet18() for _ in range(n_cnn)])

        # CNNから出力される特徴量のサイズ
        self.n_features = self.resnets[0].fc.in_features
        # 最後の層取り除く
        for resnet in self.resnets:
            resnet.fc = nn.Identity()

        self.rnn = nn.LSTM(input_size=self.n_features * n_cnn, hidden_size=self.hidden_size * n_cnn, num_layers=self.n_rnn, batch_first=True)

        #(3) FC
        self.output_layer = nn.Linear(self.hidden_size * n_cnn, n_class)

    def forward(self, input):
        # cnn features
        cnn_features =[]
        for i in range(self.n_cnn):
            cnn_feature = self.resnets[i](input[:,i])
            cnn_feature = cnn_feature.view(cnn_feature.size(0), -1)
            cnn_features.append(cnn_feature)
        cnn_features = torch.cat(cnn_features, dim=1)
        
        # LSTMに特徴量を入力
        out, _ = self.rnn(cnn_features)
        
        # 全結合層に入力し、予測値を出力
        out = self.output_layer(out)
        
        return out
    
    
class CNN(nn.Module):
    def __init__(self, n_class, n_img, pretrain):
        super(CNN, self).__init__()

        self.n_class = n_class
        self.n_img = n_img

        #(1) CNN
        # ResNet18
        #if pretrain:
        #    self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        #    for param in self.resnet.parameters():
        #        param.requires_grad = False
        #else:
        #    self.resnet = models.resnet18()
        
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
        self.output_layer = nn.Linear(num_features * n_img, n_class)

    def forward(self, input):
        # cnn features
        #print(input.size()) (batch_size, n_img, c, h, w)
        features =[]
        for i in range(self.n_img):
            feature = self.resnet(input[:,i])    # (batch_size, num_features(512))
            feature = feature.unsqueeze(1)       # (batch_size, 1, num_features(512))
            features.append(feature)
        features = torch.cat(features, dim=1)    # (batch_size, n_img, num_features(512))
        
        # FCに特徴量を入力
        out = self.output_layer(features)
        
        return out
    
class ResNet_3d_18(nn.Module):
    def __init__(self, n_class, n_img, pretrain):
        super(ResNet_3d_18, self).__init__()

        self.n_class = n_class
        self.n_img = n_img

        self.resnet = r3d_18()

    def forward(self, input):
        input = input.permute(0, 2, 1, 3, 4)
        #print(input.size())
        #print(input)
        return self.resnet(input)
    


class ResNet_3d_34(nn.Module):
    def __init__(self, n_class, n_img, pretrain):
        super(ResNet_3d_34, self).__init__()

        self.n_class = n_class
        self.n_img = n_img

        self.resnet = r3d_34()

    def forward(self, input):
        input = input.permute(0, 2, 1, 3, 4)
        #print(input.size())
        #print(input)
        return self.resnet(input)
    

class R2plus1d_18(nn.Module):
    def __init__(self, n_class, n_img, pretrain):
        super(R2plus1d_18, self).__init__()

        self.n_class = n_class
        self.n_img = n_img

        self.resnet = r2plus1d_18()

    def forward(self, input):
        input = input.permute(0, 2, 1, 3, 4)
        #print(input.size())
        #print(input)
        return self.resnet(input)
    

class R2plus1d_34(nn.Module):
    def __init__(self, n_class, n_img, pretrain):
        super(R2plus1d_34, self).__init__()

        self.n_class = n_class
        self.n_img = n_img

        self.resnet = r2plus1d_34()

    def forward(self, input):
        input = input.permute(0, 2, 1, 3, 4)
        #print(input.size())
        #print(input)
        return self.resnet(input)


class R18_LSTM(nn.Module):
    def __init__(self, n_class, n_img, hidden_size, n_rnn):
        super(R18_LSTM, self).__init__()
        
        self.n_img = n_img

        #(1) CNN
        # ResNet18
        self.resnet = models.resnet18()
        
        # CNNから出力される特徴量のサイズ(512)
        num_features = self.resnet.fc.in_features
        
        # 最後の層取り除く
        self.resnet.fc = nn.Identity()

        # LSTM
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=n_rnn, batch_first=True)

        #FC
        self.output_layer = nn.Linear(hidden_size, n_class)

    def forward(self, input):
        # cnn features
        #print(input.size()) (batch_size, n_img, c, h, w)
        cnn_features =[]
        for i in range(self.n_img):
            cnn_feature = self.resnet(input[:,i])    # (batch_size, num_features(512))
            cnn_feature = cnn_feature.unsqueeze(1)       # (batch_size, 1, num_features(512))
            cnn_features.append(cnn_feature)
        cnn_features = torch.cat(cnn_features, dim=1)    # (batch_size, n_img, num_features(512))
        
        # LSTMに特徴量を入力
        lstm_outputs, (ht, ct) = self.lstm(cnn_features)
        
        # 最後の隠れ層の出力(最後のフレームの予測)を全結合層に入力し、予測値を出力
        fc_out = self.output_layer(lstm_outputs[:, -1, :])
        
        return fc_out