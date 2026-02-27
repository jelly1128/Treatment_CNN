from torch.utils.data import Dataset, DataLoader
from .datasets import BaseMultiLabelDataset, CustomSingleLabelDataset
from .transforms import get_train_transforms, get_test_transforms

class DataLoaderFactory:
    def __init__(
            self,
            dataset_root: str,
            batch_size: int,
            num_classes: int,
            num_gpus: int = 1
        ):
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
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4*self.num_gpus,
            pin_memory=True
        )

    def create_multi_label_dataloaders(self, train_data_dirs: list, val_data_dirs: list) -> tuple[DataLoader, DataLoader]:
        """
        訓練と検証用のデータローダーを作成する

        パラメータ:
            train_splits (tuple): 訓練用データセット
            val_split (Dataset): 検証用データセット

        戻り値:
            tuple: train_loader, val_loader
        """
        # 訓練用データローダーの作成
        train_dataset = BaseMultiLabelDataset(self.dataset_root,
                                              train_data_dirs,
                                              get_train_transforms(),
                                              self.num_classes,
                                              )
        train_loader = self.create_dataloader(train_dataset, self.batch_size, shuffle=True)
        # 検証用データローダーの作成
        val_dataset = BaseMultiLabelDataset(self.dataset_root,
                                            val_data_dirs,
                                            get_test_transforms(),
                                            self.num_classes,
                                            )
        val_loader = self.create_dataloader(val_dataset, self.batch_size, shuffle=False)

        return train_loader, val_loader

    def create_multi_label_test_dataloaders(self, test_data_dirs: list) -> dict[str, DataLoader]:
        """
        テスト用のデータローダーを作成する

        パラメータ:
            splits (list): テスト用データセットのリスト

        戻り値:
            dict: テスト用データローダーの辞書
        """
        test_dataloaders = {}
        for test_data_dir in test_data_dirs:
            test_dataset = BaseMultiLabelDataset(
                self.dataset_root,
                [test_data_dir],
                get_test_transforms(),
                self.num_classes
            )
            test_dataloaders[test_data_dir] = self.create_dataloader(test_dataset, batch_size=1, shuffle=False)
        return test_dataloaders

    def create_single_label_dataloaders(self, 
                                        train_data_dirs: list, 
                                        val_data_dirs: list, 
                                        merge_label_indices: list[int], 
                                        merge_to_label: int = 4) -> tuple[DataLoader, DataLoader]:
        """
        シングルラベル用の訓練・検証データローダーを作成
        訓練データと検証データをそれぞれのディレクトリから読み込み、指定されたラベル群を1つのラベルに統一する。
        例: ラベル4,5,6,11,12をラベル4に統一
        それ以外は0~3のラベルをそのまま使用。
        Args:
            train_data_dirs (list): 訓練データのディレクトリリスト
            val_data_dirs (list): 検証データのディレクトリリスト
            merge_label_indices (list[int]): 統一するラベルのインデックスリスト
            merge_to_label (int): 統一先のラベル
        Returns:
            tuple: 訓練データローダーと検証データローダー
        """
        train_dataset = CustomSingleLabelDataset(
            self.dataset_root,
            train_data_dirs,
            get_train_transforms(),
            self.num_classes,
            merge_label_indices=merge_label_indices,
            merge_to_label=merge_to_label
        )
        train_loader = self.create_dataloader(train_dataset, self.batch_size, shuffle=True)
        val_dataset = CustomSingleLabelDataset(
            self.dataset_root,
            val_data_dirs,
            get_test_transforms(),
            self.num_classes,
            merge_label_indices=merge_label_indices,
            merge_to_label=merge_to_label
        )
        val_loader = self.create_dataloader(val_dataset, self.batch_size, shuffle=False)
        return train_loader, val_loader

    def create_single_label_test_dataloaders(self, test_data_dirs: list, merge_label_indices: list[int], merge_to_label: int = 4) -> dict[str, DataLoader]:
        """
        シングルラベル用のテストデータローダーを作成
        """
        test_dataloaders = {}
        for test_data_dir in test_data_dirs:
            test_dataset = CustomSingleLabelDataset(
                self.dataset_root,
                [test_data_dir],
                get_test_transforms(),
                self.num_classes,
                merge_label_indices=merge_label_indices,
                merge_to_label=merge_to_label
            )
            test_dataloaders[test_data_dir] = self.create_dataloader(test_dataset, batch_size=1, shuffle=False)
        return test_dataloaders