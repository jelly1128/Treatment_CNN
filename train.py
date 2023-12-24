import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn

import argparse
import os
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import pandas as pd

import dataset
import models
import results


# パラメータ
IMG_SIZE = 224
N_CLASS = 5
HIDDEN_SIZE = 128
N_RNN = 1
ROOT = 'data'
SAVE_NAME = "sample"
#MODEL_NAME = "ResNet-18"
MODEL_NAME = "r18_lstm"
#PATIENCE = 5

# k_folds  ([train], val, test)
K_FOLDS_SPLIT = [
    ([1, 2, 3, 4], 5, 6),
    ([1, 2, 3, 6], 4, 5),
    ([1, 2, 5, 6], 3, 4),
    ([1, 4, 5, 6], 2, 3),
    ([3, 4, 5, 6], 1, 2),
    ([2, 3, 4, 5], 6, 1),
]


# indexで指定された番号のfolder内のデータからデータセットを作成
def make_dataset(fold_index, image_num, olympus_data_transforms, fujifilm_data_transforms):
    olympus_videolist_file = os.path.join(ROOT, str(fold_index), 'olympus', 'olympus_subset.csv')
    olympus_img_dir = os.path.join(ROOT, str(fold_index), 'olympus', 'images')
    olympus_label_dir = os.path.join(ROOT, str(fold_index), 'olympus', 'labels')

    fujifilm_videolist_file = os.path.join(ROOT, str(fold_index), 'fujifilm', 'fujifilm_subset.csv')
    fujifilm_img_dir = os.path.join(ROOT, str(fold_index), 'fujifilm', 'images')
    fujifilm_label_dir = os.path.join(ROOT, str(fold_index), 'fujifilm', 'labels')
    
    olympus_datasets = dataset.maskedMultiDataset(olympus_videolist_file,
                                                  olympus_img_dir,
                                                  olympus_label_dir,
                                                  image_num,
                                                  system = 'olympus',
                                                  transform = olympus_data_transforms)
            
    fujifilm_datasets = dataset.maskedMultiDataset(fujifilm_videolist_file,
                                                  fujifilm_img_dir,
                                                  fujifilm_label_dir,
                                                  image_num,
                                                  system = 'fujifilm',
                                                  transform = fujifilm_data_transforms)
    
    datasets = torch.utils.data.ConcatDataset([olympus_datasets, fujifilm_datasets])
    
    return datasets


def cross_val(mode, image_num, lr, batch_size, epoch_num, pretrain):
    
    # 結果保存folderを作成
    if not os.path.exists(os.path.join(SAVE_NAME)):
        os.mkdir(os.path.join(SAVE_NAME))
    
    # デバイス
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    
    # set seed
    setup_seed(42)
    
    # transform
    olympus_data_transforms = {
        'train_val': A.Compose(
            [
                A.Crop(710,20,1890,1060),        # 外接
                #A.PadIfNeeded(1180, 1180, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                A.Resize(IMG_SIZE,IMG_SIZE),
                A.Affine(translate_percent=(0.025), rotate=(0, 5), shear=(0, 5)),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                #A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                ToTensorV2()
            ]
        ),
        'test': A.Compose(
            [
                A.Crop(710,20,1890,1060),        # 外接
                #A.PadIfNeeded(1180, 1180, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                A.Resize(IMG_SIZE,IMG_SIZE),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                #A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                ToTensorV2()
            ]
        )
    }
    
    fujifilm_data_transforms = {
        'train_val': A.Compose(
            [
                A.Crop(330,25,1590,995),         # 外接
                #A.PadIfNeeded(1260, 1260, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                A.Resize(IMG_SIZE,IMG_SIZE),
                A.Affine(translate_percent=(0.025), rotate=(0, 5), shear=(0, 5)), # 平行移動，回転，せん断
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),           # 正規化
                #A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                ToTensorV2()
            ]
        ),
        'test': A.Compose(
            [
                A.Crop(330,25,1590,995),         # 外接
                #A.PadIfNeeded(1260, 1260, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                A.Resize(IMG_SIZE,IMG_SIZE),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                #A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                ToTensorV2()
            ]
        )
    }
    
    
        
    # 実行環境をテキストファイルで保存
    with open(os.path.join(SAVE_NAME, 'Learning_environment.txt'), 'w') as f:
        f.write(f"Training mode = {mode}\n")
        f.write(f"Number of images used for prediction = {image_num}\n")
        f.write(f"Batch size = {batch_size}\n")
        f.write(f"Epoch = {epoch_num}\n")
        f.write(f"Learing rate = {lr}\n")
        f.write(f"Pretrain = {pretrain}\n")
        #f.write(f"Patience for Early Stopping = {PATIENCE}\n")
        
    print("==> Parameters: Image_num:{} LR:{} Batch_size:{}..".format(image_num, lr, batch_size))
    
    for split, (train_indices, validation_index, test_index) in enumerate(K_FOLDS_SPLIT):
        
        print(f"split = {split + 1}")
        
        # train datasets
        train_dataset_list = []
        
        for train_index in train_indices:
            train_dataset =  make_dataset(train_index, image_num, olympus_data_transforms['train_val'], fujifilm_data_transforms['train_val'])
            train_dataset_list.append(train_dataset)
        
        train_datasets = torch.utils.data.ConcatDataset(train_dataset_list)
        
        # validation datasets
        validation_datasets = make_dataset(validation_index, image_num, olympus_data_transforms['train_val'], fujifilm_data_transforms['train_val'])
        
        dataloaders = {
            'train': torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=True),
            'val': torch.utils.data.DataLoader(validation_datasets, batch_size=batch_size)
        }
        
        
        # モデル
        if MODEL_NAME == 'ResNet-18':
            model = models.CNN(n_class=N_CLASS, n_img = image_num, pretrain=pretrain)
        elif MODEL_NAME == 'ResNet-34':
            model = models.CNN(n_class=N_CLASS, n_img = image_num, pretrain=pretrain)
        elif MODEL_NAME == 'ResNet-50':
            model = models.CNN(n_class=N_CLASS, n_img = image_num, pretrain=pretrain)
        elif MODEL_NAME == '3DResNet-18':
            model = models.ResNet_3d_18(n_class=N_CLASS, n_img = image_num, pretrain=pretrain)
        elif MODEL_NAME == '3DResNet-34':
            model = models.ResNet_3d_34(n_class=N_CLASS, n_img = image_num, pretrain=pretrain)
        elif MODEL_NAME == 'r2plus1d_18':
            model = models.R2plus1d_18(n_class=N_CLASS, n_img = image_num, pretrain=pretrain)    
        elif MODEL_NAME == 'r2plus1d_34':
            model = models.R2plus1d_34(n_class=N_CLASS, n_img = image_num, pretrain=pretrain)
        elif MODEL_NAME == 'r18_lstm':
            model = models.R18_LSTM(n_class=N_CLASS, n_img = image_num, hidden_size=HIDDEN_SIZE, n_rnn=N_RNN)
        
        model = model.to(device)
        
        # 保存名
        model_name = MODEL_NAME + "_" + str(split + 1)
    
        # 学習パラメータ
        optimizer = optim.Adam(model.parameters(), lr=lr)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        
        # 学習の経過を保存
        acc_history = {'train': [], 'val': []}
        loss_history = {'train': [], 'val': []}

        save_flag = False
        counter = 0
        best_loss = 100000.0
        
        if not os.path.exists(os.path.join(SAVE_NAME, "split" + str(split + 1))):
            os.mkdir(os.path.join(SAVE_NAME, "split" + str(split + 1)))
        if not os.path.exists(os.path.join(SAVE_NAME, "split" + str(split + 1), "train")):
            os.mkdir(os.path.join(SAVE_NAME, "split" + str(split + 1), "train"))
        if not os.path.exists(os.path.join(SAVE_NAME, "split" + str(split + 1), "eval")):
            os.mkdir(os.path.join(SAVE_NAME, "split" + str(split + 1), "eval"))
        
        
        # 学習---------------------------
        print('-' * 10 + 'train' + '-' * 10)
        for epoch in range(epoch_num):
            print('-' * 10)
            print('[Epoch {}/{}]'.format(epoch+1, epoch_num))

            for phase in ['train', 'val']:

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                count = 0
                total = 0

                for batch_imgs, batch_labels in dataloaders[phase]:

                    batch_imgs = batch_imgs.to(device)
                    if mode == 'past':
                        batch_labels = batch_labels[image_num - 1].to(device)
                    if mode == 'past and future':
                        batch_labels = batch_labels[image_num // 2].to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):

                        outputs = model(batch_imgs)
                        _, predicted = torch.max(outputs, 1)
                        loss = criterion(outputs, batch_labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    #print(f"outputs = {outputs.shape}")
                    #print(f"predicted = {predicted.shape}")
                    #print(f"batch_labels = {batch_labels.shape}")
                    #print(f"loss = {loss}")
            
                    running_loss += loss.item()
                    running_corrects += torch.sum(predicted == batch_labels.data)
                    count += 1
                    total += len(batch_labels)

                # if phase == 'train':
                #     scheduler.step()

                epoch_loss = running_loss / count
                epoch_acc = (running_corrects * 100.0) / total
                acc_history[phase].append(epoch_acc.item())
                loss_history[phase].append(epoch_loss)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    
    
                # テキストファイルに保存
                with open(os.path.join(SAVE_NAME, "split" + str(split + 1), "train", 'log_' + model_name + '.txt'), 'a') as f:
                    if (phase == 'train'):
                        f.write('[Epoch {}/{}]\n'.format(epoch + 1, epoch_num))
                    f.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))
                    
                #if best_acc < epoch_acc and phase == 'val':
                #    best_acc = epoch_acc
                #    save_flag = True
                #    counter = 0
                
                # accuracyで判断
                #if phase == 'val':
                #    if best_acc < epoch_acc:
                #        best_acc = epoch_acc
                #        save_flag = True
                #        counter = 0
                #    else:
                #        counter += 1
                
                # lossで判断
                if phase == 'val':
                    if best_loss > epoch_loss:
                        best_loss = epoch_loss
                        save_flag = True
                        counter = 0
                    else:
                        counter += 1
            
            # early stopping
            #if counter >= PATIENCE:
            #    print("Early Stopping: Training stopped.")
            #    break

            # 最適なモデルの保存 
            if save_flag:
                model_save_path = os.path.join(SAVE_NAME, "split" + str(split + 1), "train", model_name + '_best' + '.pth')
                torch.save(model, model_save_path)
                print('--------model saved--------')
                save_flag = False


        # 学習結果の表示
        # accuracy
        fig1= plt.figure(figsize=(6,6))
        ax1= fig1.add_subplot()
        ax1.plot(acc_history['train'], label='train')
        ax1.plot(acc_history['val'], label='valid')
        ax1.legend()
        ax1.set_ylim(0, 100)
        fig1.savefig(os.path.join(SAVE_NAME, "split" + str(split + 1), "train", 'acc_' + model_name + '.png'))
        fig1.show()

        # loss
        fig2 = plt.figure(figsize=(6,6))
        ax2 = fig2.add_subplot()
        ax2.plot(loss_history['train'], label='train')
        ax2.plot(loss_history['val'], label='valid')
        ax2.legend()
        fig2.savefig(os.path.join(SAVE_NAME, "split" + str(split + 1), "train", 'loss_' + model_name + '.png'))
        fig2.show()
        
        # テスト---------------------------
        print('-' * 10 + 'test' + '-' * 10)
        model = torch.load(os.path.join(SAVE_NAME, "split" + str(split + 1), "train", model_name + '_best' + '.pth'))
        model.eval()
        
        test_list_csv = {
            'olympus': os.path.join(ROOT, str(test_index), 'olympus', 'olympus_subset.csv'),
            'fujifilm': os.path.join(ROOT, str(test_index), 'fujifilm', 'fujifilm_subset.csv')
        }

        test_img_dir = {
            'olympus': os.path.join(ROOT, str(test_index), 'olympus', 'images'),
            'fujifilm': os.path.join(ROOT, str(test_index), 'fujifilm', 'images')
        }

        test_label_dir = {
            'olympus': os.path.join(ROOT, str(test_index), 'olympus', 'labels'),
            'fujifilm': os.path.join(ROOT, str(test_index), 'fujifilm', 'labels')
        }
            
        test_results_csv = {
            'olympus': os.path.join(SAVE_NAME, "split" + str(split + 1), "eval", 'olympus_labels.csv'),
            'fujifilm': os.path.join(SAVE_NAME, "split" + str(split + 1), "eval", 'fujifilm_labels.csv'),
            'total': os.path.join(SAVE_NAME, "split" + str(split + 1), "eval", 'total_labels.csv')
        }
        
        test_data_transforms = {
            'olympus': olympus_data_transforms['test'],
            'fujifilm': fujifilm_data_transforms['test']
        }
        
        total_true_labels = []
        total_pred_labels = []

        for system in ['olympus', 'fujifilm']:
    
            true_labels = []
            pred_labels = []
    
            test_list = pd.read_csv(test_list_csv[system], header=None)
            test_name_list = test_list[0].values
    
            #print(test_list)
    
            for test_name in test_name_list:
        
                test_dataset = dataset.maskedTestDataset(test_name, test_img_dir[system], test_label_dir[system], image_num, system, transform=test_data_transforms[system])
                dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
        
                with torch.no_grad():
                    for batch_idx, (test_imgs, test_labels) in enumerate(dataloader):
                        test_imgs = test_imgs.to(device)
                        if mode == 'past':
                            test_labels = test_labels[image_num - 1].to(device)
                        if mode == 'past and future':
                            test_labels = test_labels[image_num // 2].to(device)
                    
                        outputs = model(test_imgs)
                        _, predicted = torch.max(outputs, 1)
                
                        true_labels.append(test_labels.item())
                        pred_labels.append(predicted.item())
                
                #print(f'test_labels_num = {len(true_labels)}')
    
            total_true_labels.extend(true_labels)
            total_pred_labels.extend(pred_labels)
    
            results.create_confusion_matrix(true_labels, pred_labels, os.path.join(SAVE_NAME, "split" + str(split + 1), "eval", system + '_cm.png'))
            results.create_csv(true_labels, pred_labels, test_results_csv[system])
    
        results.create_confusion_matrix(total_true_labels, total_pred_labels, os.path.join(SAVE_NAME, "split" + str(split + 1), "eval", 'total_cm.png'))
        results.create_csv(total_true_labels, total_pred_labels, test_results_csv['total'])
        
        
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
        

def main():
    # Parse command line arguments by argparse
    parser = argparse.ArgumentParser(
        description='Multi-class organ classification model using ResNet')
    parser.add_argument('--training_mode',
                        help='past or past and future',
                        type=str, default='past')
    parser.add_argument('--image_num',
                        help='Number of images used for prediction',
                        type=int, default=9)
    parser.add_argument('-lr', '--learning_rate',
                        help='Learning rate',
                        type=float, default=1e-4)
    parser.add_argument('-b', '--batch_size',
                        help='Batch size',
                        type=int, default=32)
    parser.add_argument('-m', '--max_epoch_num',
                        help='Maximum number of training epochs',
                        type=int, default=30)
    parser.add_argument('-p', '--pretrain',
                        help='Pretrain or not',
                        type=bool, default=False)
    
    args = parser.parse_args()
    
    cross_val(args.training_mode,
             args.image_num,
             args.learning_rate,
             args.batch_size,
             args.max_epoch_num,
             args.pretrain)
    
    
if __name__ == '__main__':
    main()