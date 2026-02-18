from dataclasses import dataclass
from enum import Enum

class DatasetType(Enum):
    """データセットの種類を表す列挙型"""
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

@dataclass
class FoldSplit:
    """
    1つのfoldにおける train/val/test の分割結果
    """
    fold_idx: int
    train: list[str]
    val: list[str]
    test: list[str]

class CVSplitter:
    """
    交差検証用のデータ分割を行うクラス
    splits_dict = {
        'split1': ['video_a', 'video_b'],
        'split2': ['video_c', 'video_d'],
        'split3': ['video_e', 'video_f'],
        'split4': ['video_g', 'video_h'],
    }
    splitter = CVSplitter(splits_dict, train_ratio=2, val_ratio=1, test_ratio=1)
    
    # イテラブルなので for で回せる
    for fold in splitter:
        print(f"Fold {fold.fold_idx}: train={fold.train}, val={fold.val}, test={fold.test}")
    
    # または特定の fold だけ取得
    fold_0 = splitter.get_fold(0)
    """
    def __init__(
        self,
        splits_dict: dict[str, list[str]],
        train_ratio: int = 2,
        val_ratio: int = 1,
        test_ratio: int = 1,
    ):
        """
        Args:
            splits_dict: YAML の cv_splits セクション（split名 → 動画フォルダ名リスト）
            train_ratio: 学習に使用する split 数
            val_ratio  : 検証に使用する split 数
            test_ratio : テストに使用する split 数
        
        Raises:
            ValueError: 分割比率の合計が split 数と一致しない場合
        """
        self.splits_dict = splits_dict
        self.split_names = list(splits_dict.keys())
        self.n_splits = len(splits_dict)
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # バリデーション
        total = train_ratio + val_ratio + test_ratio
        if total != self.n_splits:
            raise ValueError(
                f"分割比率の合計 ({total}) が split 数 ({self.n_splits}) と一致しません。"
                f"train={train_ratio}, val={val_ratio}, test={test_ratio}"
            )

    def __len__(self) -> int:
        """fold の総数を返す（= split 数）"""
        return self.n_splits
    
    def __iter__(self):
        """
        イテラブルにすることで `for fold in splitter:` が使える。
        全 fold を順に yield する。
        """
        for fold_idx in range(self.n_splits):
            yield self.get_fold(fold_idx)

    def get_fold(self, fold_idx: int) -> FoldSplit:
        """
        指定した fold の分割結果を返す。
        
        Args:
            fold_idx: fold のインデックス（0 から n_splits-1）
        
        Returns:
            FoldSplit: 型付きの分割結果
        
        計算ロジック:
            train は fold_idx から始まり train_ratio 個の split を使う
            val   は train の直後から val_ratio 個の split を使う
            test  は val の直後から test_ratio 個の split を使う
            
            すべて循環インデックスで計算するので、インデックスが n_splits を超えても
            自動的に折り返される。
        """
        if not 0 <= fold_idx < self.n_splits:
            raise IndexError(f"fold_idx={fold_idx} は範囲外です（0〜{self.n_splits-1}）")
        
        # 循環インデックスで train/val/test の split インデックスを取得
        train_indices = self._circular_indices(fold_idx, self.train_ratio)
        val_indices = self._circular_indices(fold_idx + self.train_ratio, self.val_ratio)
        test_indices = self._circular_indices(fold_idx + self.train_ratio + self.val_ratio, self.test_ratio)
        
        # 各 split に含まれる動画フォルダ名を収集
        train_folders = self._collect_folders(train_indices)
        val_folders = self._collect_folders(val_indices)
        test_folders = self._collect_folders(test_indices)
        
        return FoldSplit(fold_idx, train_folders, val_folders, test_folders)
    
    def _circular_indices(self, start: int, size: int) -> list[int]:
        """
        循環インデックスを生成する。
        
        Args:
            start: 開始インデックス
            size : 取得する個数
        
        Returns:
            循環インデックスのリスト
        
        例:
            n_splits=4 のとき
            _circular_indices(0, 2) → [0, 1]
            _circular_indices(3, 2) → [3, 0]  # 折り返す
            _circular_indices(5, 2) → [1, 2]  # start も折り返す
        """
        return [(start + i) % self.n_splits for i in range(size)]
    
    def _collect_folders(self, indices: list[int]) -> list[str]:
        """
        指定したインデックスの split に含まれる全動画フォルダを収集する。
        
        Args:
            indices: split のインデックスリスト
        
        Returns:
            動画フォルダ名のリスト
        """
        folders = []
        for idx in indices:
            split_name = self.split_names[idx]
            folders.extend(self.splits_dict[split_name])
        return folders

# ──────────────────────────────────────────
# 補助関数
# ──────────────────────────────────────────

def print_fold_summary(splitter: CVSplitter) -> None:
    """
    全 fold の分割内容を表示する補助関数。
    デバッグ・確認用。
    """
    print(f"交差検証設定: {len(splitter)} folds")
    print(f"分割比率: train={splitter.train_ratio}, val={splitter.val_ratio}, test={splitter.test_ratio}")
    print(f"Split 名: {splitter.split_names}\n")
    
    for fold in splitter:
        print(f"Fold {fold.fold_idx}:")
        
        # どの split を使っているか表示
        train_splits = [splitter.split_names[i] for i in splitter._circular_indices(fold.fold_idx, splitter.train_ratio)]
        val_splits = [splitter.split_names[i] for i in splitter._circular_indices(fold.fold_idx + splitter.train_ratio, splitter.val_ratio)]
        test_splits = [splitter.split_names[i] for i in splitter._circular_indices(fold.fold_idx + splitter.train_ratio + splitter.val_ratio, splitter.test_ratio)]
        
        print(f"  Train splits: {train_splits}")
        print(f"  Val splits  : {val_splits}")
        print(f"  Test splits : {test_splits}")
        print(f"  Train folders: {len(fold.train)} videos")
        for folder in fold.train:
            print(f"    - {folder}")
        print(f"  Val folders  : {len(fold.val)} videos")
        for folder in fold.val:
            print(f"    - {folder}")
        print(f"  Test folders : {len(fold.test)} videos")
        for folder in fold.test:
                    print(f"    - {folder}")
        print()