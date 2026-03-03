import numpy as np
from evaluate.result_types import HardMultiLabelResult, SingleLabelResult
from utils.window_key import WindowSizeKey

class Analyzer:
    def __init__(self, num_classes: int):
        """
        推論結果を解析するクラス。

        Args:
            num_classes (int): クラス数
        """
        self.num_classes = num_classes

    def apply_sliding_window_to_hard_multi_label_results(self, hard_multi_label_results: dict[str, HardMultiLabelResult], window_size=5, step=1):
        """
        マルチラベルの予測結果にスライディングウィンドウを適用して、主クラス（0-5）のシングルラベル予測に変換する関数

        Args:
            hard_multi_label_results (dict[str, HardMultiLabelResult]): 各フォルダのマルチラベルの結果
            window_size (int): スライディングウィンドウのサイズ
            step (int): スライディングウィンドウのステップ幅

        Returns:
            dict[str, SingleLabelResult]: 各フォルダの主クラスのシングルラベル予測結果
        """
        smoothed_results = {}

        for folder_name, result in hard_multi_label_results.items():
            y_pred = np.array(result.multi_labels)[:, :self.num_classes]  # 主クラスのみ
            y_true = np.array(result.ground_truth_labels)[:, :self.num_classes]  # 主クラスのみ
            num_frames = len(y_pred)

            smoothed_labels = []
            center_indices = []

            for start in range(0, num_frames - window_size + 1, step):
                window_pred = y_pred[start:start + window_size]
                class_counts_pred = window_pred.sum(axis=0)
                smoothed_label = np.argmax(class_counts_pred)

                center = start + window_size // 2
                smoothed_labels.append(smoothed_label)
                center_indices.append(center)

            # 中心インデックスに対応する画像パスと正解ラベルを取得
            image_paths_centered = [result.image_paths[i] for i in center_indices]
            true_labels = [np.argmax(y_true[i]) for i in center_indices]

            smoothed_results[folder_name] = SingleLabelResult(
                image_paths=image_paths_centered,
                single_labels=smoothed_labels,
                ground_truth_labels=true_labels
            )

        return smoothed_results

    def analyze_sliding_windows(self,
                                hard_multi_label_results: dict[str, HardMultiLabelResult],
                                window_sizes: list=None):
        """
        異なるウィンドウサイズでスライディングウィンドウ解析を実行し、結果をまとめる

        Args:
            hard_multi_label_results: マルチラベルの予測結果
            window_sizes: ウィンドウサイズのリスト（Noneの場合はデフォルト値を使用）

        Returns:
            dict: 各ウィンドウサイズの結果
        """
        if window_sizes is None:
            window_sizes = range(3, 16, 2)  # 3, 5, 7, 9, 11, 13, 15

        # 結果保存用の辞書を初期化
        all_window_results = WindowSizeKey.initialize_results(window_sizes)

        for window_size in window_sizes:
            window_key = WindowSizeKey.create(window_size)
            # スライディングウィンドウを適用
            sliding_window_results = self.apply_sliding_window_to_hard_multi_label_results(
                hard_multi_label_results,
                window_size=window_size
            )
            all_window_results[window_key] = sliding_window_results

        return all_window_results