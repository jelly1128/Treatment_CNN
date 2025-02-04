from torch.utils.data import DataLoader, ConcatDataset
from .datasets import MultiLabelDetectionDataset, MultiLabelDetectionDatasetForTest
from .transforms import get_train_transforms, get_test_transforms

def create_multilabel_train_dataloaders(config, fold, num_gpus):
    """
    指定されたフォールドのデータローダーを作成します
    
    パラメータ:
        config (dataclass): 設定
        fold (tuple): 3つの文字列からなるタプル、それぞれの文字列はスプリットのビデオIDです
    
    戻り値:
        tuple: train_loader, val_loader
    """
    train_splits = [MultiLabelDetectionDataset(config.paths.root,
                                               transform=get_train_transforms(),
                                               num_classes=config.training.num_classes,
                                               split=split) 
                    for split in fold[:2]]
    train_dataset = ConcatDataset(train_splits)
    val_split = MultiLabelDetectionDataset(config.paths.root,
                                           transform=get_test_transforms(),
                                           num_classes=config.training.num_classes,
                                           split=fold[2])
    
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size * num_gpus, shuffle=True, num_workers=4 * num_gpus)
    val_loader = DataLoader(val_split, batch_size=config.training.batch_size * num_gpus, shuffle=False, num_workers=4 * num_gpus)
    
    return train_loader, val_loader


def create_multilabel_test_dataloaders(config, split, num_gpus):
    test_dataloaders = {}
    for folder_name in split:
        # # 結果保存用フォルダを作成
        # save_path = os.path.join(config.paths.save_dir, folder_name)
        # os.makedirs(save_path, exist_ok=True)

        # データセット作成
        test_dataset = MultiLabelDetectionDatasetForTest(
            config.paths.root,
            folder_name,
            get_test_transforms(),
            config.test.num_classes
        )

        # データローダー作成
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4 * num_gpus
        )

        test_dataloaders[folder_name] = test_dataloader
    
    return test_dataloaders