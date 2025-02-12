from torch.utils.data import Dataset, DataLoader, ConcatDataset
from .datasets import MultiLabelDetectionDataset, MultiLabelDetectionDatasetForTest
from .transforms import get_train_transforms, get_test_transforms

def create_multilabel_train_dataloaders(config, num_gpus, train_data):
    """
    指定されたフォールドのデータローダーを作成します
    
    パラメータ:
        config (dataclass): 設定
        fold (tuple): 3つの文字列からなるタプル、それぞれの文字列はsplitのビデオIDです
    
    戻り値:
        tuple: train_loader, val_loader
    """
    train_splits = [MultiLabelDetectionDataset(config.paths.dataset_root,
                                               transform=get_train_transforms(),
                                               num_classes=config.training.num_classes,
                                               split=split) 
                    for split in fold[:2]]
    train_dataset = ConcatDataset(train_splits)
    val_split = MultiLabelDetectionDataset(config.paths.dataset_root,
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
            config.paths.dataset_root,
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


class DataLoaderFactory:
    def __init__(self, dataset_root: str, batch_size: int, num_classes: int, num_gpus: int):
        """
        データローダーファクトリの初期化

        パラメータ:
            dataset_root (str): データセットが格納されているディレクトリのパス
            batch_size (int): バッチサイズ
            num_classes (int): クラス数
            num_gpus (int): GPUの数
        """
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_gpus = num_gpus

    def create_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
        """
        汎用データローダー生成メソッド

        パラメータ:
            dataset (Dataset): データセット
            batch_size (int): バッチサイズ
            shuffle (bool): シャッフルするかどうか

        戻り値:
            DataLoader: データローダー
        """
        return DataLoader(
            dataset,
            batch_size=batch_size * self.num_gpus,
            shuffle=shuffle,
            num_workers=4 * self.num_gpus
        )

    def create_train_val_dataloaders(self, train_data_dirs: list, val_data_dirs: tuple):
        """
        訓練と検証用のデータローダーを作成する

        パラメータ:
            train_splits (tuple): 訓練用データセット
            val_split (Dataset): 検証用データセット

        戻り値:
            tuple: train_loader, val_loader
        """
        train_data_paths = []
        for data_dir in train_data_dirs:
            train_data_
        
        # 訓練用データローダーの作成
        train_dataset = MultiLabelDetectionDataset()
        train_loader = self.create_dataloader(train_data_dirs, self.batch_size, shuffle=True)
        # 検証用データローダーの作成
        val_dataset = MultiLabelDetectionDataset(
            self.dataset_root,
            val_data_dir,
            get_test_transforms(),
            self.num_classes
        )
        val_loader = self.create_dataloader(val_dataset, self.batch_size, shuffle=False)

        return train_loader, val_loader

    def create_test_dataloaders(self, splits):
        """
        テスト用のデータローダーを作成する

        パラメータ:
            splits (list): テスト用データセットのリスト

        戻り値:
            dict: テスト用データローダーの辞書
        """
        test_dataloaders = {}
        for split in splits:
            test_dataset = MultiLabelDetectionDatasetForTest(
                self.dataset_root,
                split,
                get_test_transforms(),
                self.num_classes
            )
            test_dataloaders[split] = self.create_dataloader(test_dataset, batch_size=1, shuffle=False)
        return test_dataloaders

# 使用例
# config = ...  # 設定を読み込む
# num_gpus = 2  # GPUの数

# # configから必要な情報を抽出
# dataset_root = config.paths.dataset_root
# batch_size = config.training.batch_size
# num_classes = config.training.num_classes

# # ファクトリのインスタンスを作成
# factory = DataLoaderFactory(dataset_root, batch_size, num_classes, num_gpus)

# # 訓練と検証用データローダーの作成
# train_splits = [MultiLabelDetectionDataset(dataset_root,
#                                           transform=get_train_transforms(),
#                                           num_classes=num_classes,
#                                           split=split) 
#                 for split in fold[:2]]
# val_split = MultiLabelDetectionDataset(dataset_root,
#                                       transform=get_test_transforms(),
#                                       num_classes=num_classes,
#                                       split=fold[2])
# train_loader, val_loader = factory.create_train_val_dataloaders(train_splits, val_split)

# # テスト用データローダーの作成
# test_dataloaders = factory.create_test_dataloaders(test_splits)