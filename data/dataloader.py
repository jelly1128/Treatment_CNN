from torch.utils.data import DataLoader, ConcatDataset
from .datasets import MultiLabelDetectionDataset
from .transforms import get_train_transforms, get_test_transforms

def create_train_val_dataloaders(config, fold, num_gpus):
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