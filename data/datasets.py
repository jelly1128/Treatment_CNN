import os
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, List, Dict
import csv
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

TREATMENT_CLASS = 4

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, anomaly_detector = True):
        """
        カスタムデータセットの初期化
        
        :param root_dir: データセットのルートディレクトリ
        :param transform: 画像変換用の関数（オプション）
        :param anomaly_detector: 異常検出モードのフラグ（デフォルトはTrue）
        """
        self.root_dir = root_dir
        self.img_dict = {}
        self.transform = transform
        
        # csv_files = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith('.csv')]
        # print(csv_files)
        
        # サブフォルダ
        self.subfolder_names = [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
        print(self.subfolder_names)
        # for csv_file in csv_files:
        for subfolder_name in self.subfolder_names:
            with open(f'{root_dir}/{subfolder_name}.csv', mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    label = int(row[1])
                    if anomaly_detector:
                        self.img_dict[os.path.join(subfolder_name, row[0])] = label
                    elif label < TREATMENT_CLASS:
                        self.img_dict[os.path.join(subfolder_name, row[0])] = label
                    # print(os.path.join(subfolder_name, row[0]))

    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, idx):
        img_names = list(self.img_dict.keys())
        # print(img_names)
        img_path = os.path.join(self.root_dir, img_names[idx])
        label = self.img_dict[img_names[idx]]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_names[idx], label


class BaseDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        """
        基本的なデータセットクラスの初期化
        
        :param root_dir: データセットのルートディレクトリ
        :param transform: 画像変換用の関数（オプション）
        """
        self.root_dir = root_dir
        self.img_dict = {}               # 画像パスとラベルを格納する辞書
        self.transform = transform
        
        # ルートディレクトリ下のサブフォルダ名のリストを取得
        self.subfolder_names = [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
        print(self.subfolder_names)
        
        self._load_data()

    def _load_data(self):
        """
        データの読み込み（子クラスでオーバーライド）
        """
        pass

    def __len__(self):
        """データセットの長さを返す"""
        return len(self.img_dict)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str, int]:
        """
        指定されたインデックスのデータ項目を取得
        
        :param idx: データ項目のインデックス
        :return: (画像テンソル, 画像パス, ラベル)のタプル
        """
        img_names = list(self.img_dict.keys())
        img_path = os.path.join(self.root_dir, img_names[idx])
        label = self.img_dict[img_names[idx]]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_names[idx], label
    
    
class AnomalyDetectionDataset(BaseDataset):
    def __init__(self, root_dir: str, transform, num_classes: int):
        self.num_classes = num_classes
        super().__init__(root_dir, transform)
        
    def _load_data(self):
        """
        異常検出用のデータ読み込み
        """
        # 各サブフォルダに対応するcsvファイルを読み込み，画像パスとラベルを辞書に格納する
        if self.num_classes == 2:  # 2値分類（正常/異常）
            for subfolder_name in self.subfolder_names:
                with open(f'{self.root_dir}/{subfolder_name}.csv', mode='r') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        original_label = int(row[1])
                        # 4未満は0、4以上は1に変更
                        label = 0 if original_label < TREATMENT_CLASS else 1
                        self.img_dict[os.path.join(subfolder_name, row[0])] = label
                        
        elif self.num_classes == 4: # 正常4クラス(時系列無し処置分類)
            # 各サブフォルダに対応するcsvファイルを読み込み，画像パスとラベルを辞書に格納する
            for subfolder_name in self.subfolder_names:
                with open(f'{self.root_dir}/{subfolder_name}.csv', mode='r') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        label = int(row[1])
                        # 異常クラス(4~)なら4に統一
                        if label < TREATMENT_CLASS:
                            self.img_dict[os.path.join(subfolder_name, row[0])] = label
                            
        elif self.num_classes == 5: # 正常4クラス/異常1クラス
            # 各サブフォルダに対応するcsvファイルを読み込み，画像パスとラベルを辞書に格納する
            for subfolder_name in self.subfolder_names:
                with open(f'{self.root_dir}/{subfolder_name}.csv', mode='r') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        label = int(row[1])
                        # 異常クラス(4~)なら4に統一
                        if label >= TREATMENT_CLASS:
                            label = 4
                        self.img_dict[os.path.join(subfolder_name, row[0])] = label
                        
        else:
            for subfolder_name in self.subfolder_names:
                with open(f'{self.root_dir}/{subfolder_name}.csv', mode='r') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        label = int(row[1])
                        self.img_dict[os.path.join(subfolder_name, row[0])] = label
                    

class TreatmentClassificationDataset(BaseDataset):
    def __init__(self, root_dir: str, n_image: int, transform=None):
        """
        時系列データセットクラスの初期化
        
        :param root_dir: データセットのルートディレクトリ
        :param n_image: 1つのサンプルに含まれる画像の枚数
        :param transform: 画像変換用の関数（オプション）
        """
        super().__init__(root_dir, transform)
        self.n_image = n_image
        self.dataset = self.create_dataset()
        self.label_counts = self.get_label_counts()
        
    def _load_data(self):
        """
        処置分類用のデータ読み込み
        """
        # 各サブフォルダに対応するcsvファイルを読み込み，画像パスとラベルを辞書に格納する
        for subfolder_name in self.subfolder_names:
            with open(f'{self.root_dir}/{subfolder_name}.csv', mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    label = int(row[1])
                    if label < TREATMENT_CLASS:
                        self.img_dict[os.path.join(subfolder_name, row[0])] = label
                        
    def create_dataset(self) -> List[Tuple[Tuple[str, ...], int]]:
        """
        n_image枚ずつのサンプルを作成する

        Returns:
            List[Tuple[Tuple[str, ...], int]]: Tuple[Tuple[画像パス, ...], ラベル]のリスト
        """
        dataset = []
        img_paths = list(self.img_dict.keys())
        current_label = None
        current_sample = []

        for img_path in img_paths:
            label = self.img_dict[img_path]
            if current_label is None or label == current_label:
                current_sample.append(img_path)
                if len(current_sample) == self.n_image:
                    dataset.append((tuple(current_sample), label))
                    current_sample = []
                    current_label = label
            else:
                current_sample = [img_path]
                current_label = label

        return dataset
    
    def get_label_counts(self) -> Dict[int, int]:
        """
        ラベルごとのデータ数を取得する
        
        Returns:
            Dict[int, int]: ラベルをキー、データ数を値とする辞書
        """
        label_counts = {}
        for _, label in self.dataset:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        return label_counts
    
    def __len__(self):
        """データセットの長さを返す"""
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Tuple[str, ...], torch.Tensor]:
        """
        指定されたインデックスのデータ項目を取得
        Returns:
            Tuple[torch.Tensor, Tuple[str, ...], torch.Tensor]: 画像テンソル、画像パスのタプル、ラベルテンソル
        """
        img_paths, label = self.dataset[idx]
        images = [Image.open(os.path.join(self.root_dir, img_path)).convert("RGB") for img_path in img_paths]

        if self.transform:
            images = [self.transform(image) for image in images]

        return torch.stack(images), img_paths, label
    
    def save_images_by_label(self, output_dir: str, max_images_per_label: int = 100):
        """
        ラベルごとに画像をまとめて保存する

        :param output_dir: 出力ディレクトリ
        :param max_images_per_label: 各ラベルで保存する最大画像数
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # ラベルごとに画像を集める
        images_by_label: Dict[int, List[torch.Tensor]] = {}
        for idx in range(len(self)):
            images, _, label = self[idx]
            if label not in images_by_label:
                images_by_label[label] = []
            images_by_label[label].extend(images)

        # ラベルごとに画像を保存
        for label, images in images_by_label.items():
            # 最大画像数に制限
            images = images[:max_images_per_label]
            
            # 画像をグリッドに配置
            grid = make_grid(images, nrow=int(np.sqrt(len(images))))
            
            # 画像を保存
            plt.figure(figsize=(20, 20))
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis('off')
            plt.title(f'Label: {label}')
            plt.savefig(os.path.join(output_dir, f'label_{label}.png'))
            plt.close()

        print(f"Images saved in {output_dir}")


class BaseDatasetForTest(Dataset):
    def __init__(self, root_path: str, transform=None):
        """
        基本的なテスト用データセットクラスの初期化
        
        :param root_path: データセットのルートディレクトリ
        :param transform: 画像変換用の関数（オプション）
        """
        self.root_path = root_path
        self.img_dict = {}               # 画像パスとラベルを格納する辞書
        
        self._load_data()

    def _load_data(self):
        """
        データの読み込み（子クラスでオーバーライド）
        """
        pass

    def __len__(self):
        """データセットの長さを返す"""
        return len(self.img_dict)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str, int]:
        """
        指定されたインデックスのデータ項目を取得
        
        :param idx: データ項目のインデックス
        :return: (画像テンソル, 画像パス, ラベル)のタプル
        """
        img_names = list(self.img_dict.keys())
        img_path = os.path.join(self.root_dir, img_names[idx])
        label = self.img_dict[img_names[idx]]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_names[idx], label

    
class AnomalyDetectionDatasetForTest(Dataset):
    def __init__(self, root_path: str, test_dir: str, transform, num_classes: int):
        
        """
        基本的なテスト用データセットクラスの初期化
        
        :param root_path: データセットのルートディレクトリ    ex) data/test/olympus
        :param transform: 画像変換用の関数（オプション）
        """
        self.test_dir_path = os.path.join(root_path, test_dir)
        self.img_dict = {}               # 画像ファイル名とラベルを格納する辞書
        self.transform = transform
        # print(f"test_dir_path {self.test_dir_path}")
        """
        異常検出用のデータ読み込み
        """
        # 各サブフォルダに対応するcsvファイルを読み込み，画像ファイル名とラベルを辞書に格納する
        if num_classes == 2:  # 2値分類（正常/異常）
            with open(f'{root_path}/{test_dir}.csv', mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    original_label = int(row[1])
                    # 4未満は0、4以上は1に変更
                    label = 0 if original_label < TREATMENT_CLASS else 1
                    self.img_dict[row[0]] = label
                        
        elif num_classes == 4: # 正常4クラス(時系列無し処置分類)
            # 各サブフォルダに対応するcsvファイルを読み込み，画像パスとラベルを辞書に格納する
            with open(f'{root_path}/{test_dir}.csv', mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    label = int(row[1])
                    # 異常クラス(4~)なら4に統一
                    if label < TREATMENT_CLASS:
                        self.img_dict[row[0]] = label
                            
        elif num_classes == 5: # 正常4クラス/異常1クラス
            # 各サブフォルダに対応するcsvファイルを読み込み，画像パスとラベルを辞書に格納する
            with open(f'{root_path}/{test_dir}.csv', mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    label = int(row[1])
                    # 異常クラス(4~)なら4に統一
                    if label >= TREATMENT_CLASS:
                        label = 4
                    self.img_dict[row[0]] = label            
        
        else:
            with open(f'{root_path}/{test_dir}.csv', mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    label = int(row[1])
                    self.img_dict[row[0]] = label
    
    def __len__(self):
        """データセットの長さを返す"""
        return len(self.img_dict)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, str, int]:
        """
        指定されたインデックスのデータ項目を取得
        
        :param idx: データ項目のインデックス
        :return: (画像テンソル, 画像パス, ラベル)のタプル
        """
        img_names = list(self.img_dict.keys())
        # img_path = img_names[idx]
        img_path = os.path.join(self.test_dir_path, img_names[idx])
        label = self.img_dict[img_names[idx]]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_names[idx], label


class TreatmentClassificationDatasetForTest(Dataset):
    def __init__(self, root_dir: str, data: list, n_image: int, transform):
        """
        時系列データセットクラスの初期化
        
        """
        self.root_dir = root_dir
        self.data = data
        self.n_image = n_image
        self.transform = transform
        
    def __len__(self):
        return len(self.data) - self.n_image + 1
    
    def __getitem__(self, idx):
        image_paths = [self.data[idx + i][0] for i in range(self.n_image)]
        ground_truth = self.data[idx + self.n_image - 1][1]
        images = [Image.open(os.path.join(self.root_dir, image_path)).convert("RGB") for image_path in image_paths]
        images = [self.transform(image) for image in images]
        
        return torch.stack(images, dim=0), image_paths[-1], ground_truth   # n_image枚の画像とn_image枚目のパスとラベル


class MultiLabelDetectionDataset(Dataset):
    def __init__(self, root_dir: str, transform: callable, num_classes: int, split: tuple) -> None:
        """
        Initialize a basic dataset class for testing purposes.

        Args:
            root_dir (str): The root directory of the dataset.
            transform (Callable): An optional function to transform the images.
            num_classes (int): The number of classes in the dataset.
            split (tuple): A tuple of strings representing the names of the subfolders
                in the root directory.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes
        self.split = split

        self.img_dict = {}  # A dictionary to store image file names and labels.

        # Read the CSV files in each subfolder and store the image file names and labels.
        for subfolder in self.split:
            with open(os.path.join(self.root_dir, subfolder + '.csv'), 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    labels = [int(label) for label in row[1:] if label.isdigit()]

                    if self.num_classes == 6:
                        # Use only the labels from 0 to 5.
                        labels = [label for label in labels if 0 <= label <= 5]
                    elif self.num_classes == 7:
                        # Replace the labels from 6 to 14 with 6.
                        labels = [6 if 6 <= label <= 14 else label for label in labels]

                    labels.sort()  # Sort the labels in ascending order.
                    self.img_dict[os.path.join(subfolder, row[0])] = labels
    
    def __len__(self) -> int:
        """データセットの長さを返す"""
        return len(self.img_dict)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, str, int]:
        """
        指定されたインデックスのデータ項目を取得
        
        :param idx: データ項目のインデックス
        :return: (画像テンソル, 画像パス, ラベル)のタプル
        """
        img_names = list(self.img_dict.keys())
        img_path = os.path.join(self.root_dir, img_names[idx])
        labels = self.img_dict[img_names[idx]]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            
        # ラベルをone-hotエンコード形式に変換
        one_hot_label = torch.zeros(self.num_classes)  # num_classes次元のzeroベクトルを作成
        for label in labels:
            one_hot_label[label] = 1  # 該当するラベルのインデックスを1にセット
            
        # print(img_names[idx], one_hot_label)

        return image, img_names[idx], one_hot_label
    
    
class MultiLabelDetectionDatasetForTest(Dataset):
    def __init__(self, root_path: str, test_dir: str, transform, num_classes: int):
        
        """
        基本的なテスト用データセットクラスの初期化
        
        :param root_path: データセットのルートディレクトリ    ex) data/test/olympus
        :param transform: 画像変換用の関数（オプション）
        """
        self.test_dir_path = os.path.join(root_path, test_dir)
        self.img_dict = {}               # 画像ファイル名とラベルを格納する辞書
        self.transform = transform
        self.num_classes = num_classes
        # print(f"test_dir_path {self.test_dir_path}")
        
        with open(f'{root_path}/{test_dir}.csv', mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                labels = [int(label) for label in row[1:] if label.strip().isdigit()]
                    
                if num_classes == 6:
                # 0～5のクラスデータのみを使用
                    labels = [label for label in labels if 0 <= label <= 5]
                    
                elif num_classes == 7:
                # 無効フレームのラベルをすべて6に置き換える
                    labels = [6 if 6 <= label <= 14 else label for label in labels]
                    
                labels.sort()  # 昇順にソート
                # print(row, labels)
                self.img_dict[os.path.join(row[0])] = labels
                
                """
                前コード
                """
                # # 処置4クラス+体外2クラス(+無効フレーム9クラス)
                # labels = [int(label) for label in row[1:] if label.strip().isdigit()]
                # labels.sort()  # ここで昇順にソート
                # # print(row, labels)
                # self.img_dict[os.path.join(row[0])] = labels
                
                # 処置4クラス+体外2クラス(+無効フレーム1クラス)(5,6入れ替え後)
                # labels = [int(label) for label in row[1:] if label.strip().isdigit()]
                # # # 無効フレームのラベルをすべて6に置き換える
                # labels = [6 if 6 <= label <= 14 else label for label in labels]
                # labels.sort()  # ここで昇順にソート
                # self.img_dict[os.path.join(row[0])] = labels
                
                # 処置4クラス+体外2クラス(5,6入れ替え後)
                # labels = [int(label) for label in row[1:] if label.strip().isdigit()]
                # filtered_labels = [label for label in labels if 0 <= label <= 5]
                # filtered_labels.sort()  # 昇順にソート
                # # print(row, filtered_labels)
                # self.img_dict[os.path.join(row[0])] = filtered_labels
    
    def __len__(self):
        """データセットの長さを返す"""
        return len(self.img_dict)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, str, int]:
        """
        指定されたインデックスのデータ項目を取得
        
        :param idx: データ項目のインデックス
        :return: (画像テンソル, 画像パス, ラベル)のタプル
        """
        img_names = list(self.img_dict.keys())
        # img_path = img_names[idx]
        img_path = os.path.join(self.test_dir_path, img_names[idx])
        labels = self.img_dict[img_names[idx]]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            
        # ラベルをone-hotエンコード形式に変換
        one_hot_label = torch.zeros(self.num_classes)  # num_classes次元のzeroベクトルを作成
        for label in labels:
            one_hot_label[label] = 1  # 該当するラベルのインデックスを1にセット
            
        # print(img_names[idx], one_hot_label)

        return image, img_names[idx], one_hot_label
    