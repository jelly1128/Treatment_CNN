import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats

class EntropyDistributionAnalyzer:
    def __init__(self, save_dir_path: Path):
        """
        エントロピーの分布を解析するクラス
        
        Args:
            save_dir_path (Path): 結果を保存するディレクトリパス
        """
        self.save_dir_path = save_dir_path
        self.save_dir_path.mkdir(parents=True, exist_ok=True)
        
    def analyze_entropy_distribution(self, entropies_list: list[float], video_name: str = ""):
        """
        エントロピーの分布を解析する
        
        Args:
            entropies_list (list[float]): エントロピー値のリスト
            video_name (str, optional): 解析対象の動画名
        """
        # 基本統計量の計算
        stats_data = self._calculate_basic_statistics(entropies_list)
        
        # 分布の可視化
        self._visualize_distribution(entropies_list, video_name)
        
        # エントロピーの時系列変化率を計算
        if len(entropies_list) > 1:
            self._analyze_entropy_changes(entropies_list, video_name)
        
        # 異常値の検出
        outliers = self._detect_outliers(entropies_list)
        
        # クラスタリング
        clusters = self._cluster_entropy_values(entropies_list)
        
        return {
            "basic_stats": stats_data,
            "outliers": outliers,
            "clusters": clusters
        }
    
    def _calculate_basic_statistics(self, entropies_list: list[float], video_name: str = ""):
        """基本統計量を計算する"""
        stats_data = {
            "mean": np.mean(entropies_list),
            "median": np.median(entropies_list),
            "std": np.std(entropies_list),
            "min": np.min(entropies_list),
            "max": np.max(entropies_list),
            "q1": np.percentile(entropies_list, 25),
            "q3": np.percentile(entropies_list, 75),
            "skewness": stats.skew(entropies_list),
            "kurtosis": stats.kurtosis(entropies_list)
        }
        
        # 統計情報をCSVに保存
        stats_df = pd.DataFrame([stats_data])
        stats_df.to_csv(self.save_dir_path / f"{video_name}_entropy_stats.csv", index=False)
        
        return stats_data
    
    def _visualize_distribution(self, entropies_list: list[float], video_name: str):
        """エントロピーの分布を可視化する"""
        plt.figure(figsize=(12, 10))
        
        # サブプロット1: ヒストグラム
        plt.subplot(2, 2, 1)
        sns.histplot(entropies_list, kde=True, bins=30)
        plt.title(f'Entropy Distribution - {video_name}')
        plt.xlabel('Normalized Entropy')
        plt.ylabel('Frequency')
        
        # サブプロット2: カーネル密度推定
        plt.subplot(2, 2, 2)
        sns.kdeplot(entropies_list, fill=True)
        plt.title(f'Entropy Density - {video_name}')
        plt.xlabel('Normalized Entropy')
        
        # サブプロット3: 箱ひげ図
        plt.subplot(2, 2, 3)
        sns.boxplot(y=entropies_list)
        plt.title(f'Entropy Boxplot - {video_name}')
        plt.ylabel('Normalized Entropy')
        
        # サブプロット4: ECDF (経験的累積分布関数)
        plt.subplot(2, 2, 4)
        sns.ecdfplot(entropies_list)
        plt.title(f'Empirical CDF - {video_name}')
        plt.xlabel('Normalized Entropy')
        plt.ylabel('Cumulative Probability')
        
        plt.tight_layout()
        plt.savefig(self.save_dir_path / f"{video_name}_entropy_distribution.png", dpi=300)
        plt.close()
        
        # エントロピー値の時系列プロット
        plt.figure(figsize=(14, 6))
        plt.plot(range(len(entropies_list)), entropies_list)
        plt.title(f'Entropy Time Series - {video_name}')
        plt.xlabel('Frame Index')
        plt.ylabel('Normalized Entropy')
        plt.grid(True, alpha=0.3)
        
        # 移動平均を追加
        # window_size = min(50, len(entropies_list) // 10) if len(entropies_list) > 10 else 1
        window_size = 11
        if window_size > 0:
            moving_avg = np.convolve(entropies_list, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, window_size-1+len(moving_avg)), moving_avg, 'r-', label=f'Moving Avg (window={window_size})')
            plt.legend()
        
        plt.savefig(self.save_dir_path / f"{video_name}_entropy_timeseries.png", dpi=300)
        plt.close()
    
    def _analyze_entropy_changes(self, entropies_list: list[float], video_name: str):
        """エントロピーの変化率を計算し、急激な変化を検出する"""
        # 連続するフレーム間のエントロピー変化率を計算
        entropy_changes = np.diff(entropies_list)
        
        # 変化率の絶対値が大きい上位5%のフレームを抽出
        threshold = np.percentile(np.abs(entropy_changes), 95)
        significant_changes = np.where(np.abs(entropy_changes) > threshold)[0]
        
        # 変化率のヒストグラム
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        sns.histplot(entropy_changes, kde=True, bins=30)
        plt.title(f'Entropy Changes Distribution - {video_name}')
        plt.xlabel('Frame-to-Frame Entropy Change')
        plt.ylabel('Frequency')
        
        # 変化率の時系列プロット
        plt.subplot(2, 1, 2)
        plt.plot(range(len(entropy_changes)), entropy_changes)
        plt.scatter(significant_changes, entropy_changes[significant_changes], color='red', label='Significant Changes')
        plt.title(f'Entropy Changes Over Time - {video_name}')
        plt.xlabel('Frame Index')
        plt.ylabel('Entropy Change')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir_path / f"{video_name}_entropy_changes.png", dpi=300)
        plt.close()
        
        # 重要な変化点の情報をCSVに保存
        if len(significant_changes) > 0:
            significant_df = pd.DataFrame({
                'frame_index': significant_changes + 1,  # 1-indexed for readability
                'previous_frame': significant_changes,
                'entropy_change': entropy_changes[significant_changes],
                'previous_entropy': [entropies_list[i] for i in significant_changes],
                'current_entropy': [entropies_list[i+1] for i in significant_changes]
            })
            significant_df.to_csv(self.save_dir_path / f"{video_name}_significant_changes.csv", index=False)
    
    def _detect_outliers(self, entropies_list: list[float]):
        """IQR法を用いて外れ値を検出する"""
        q1 = np.percentile(entropies_list, 25)
        q3 = np.percentile(entropies_list, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers_indices = np.where((entropies_list < lower_bound) | (entropies_list > upper_bound))[0]
        outliers_values = [entropies_list[i] for i in outliers_indices]
        
        return {
            "indices": outliers_indices,
            "values": outliers_values,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
    
    def _cluster_entropy_values(self, entropies_list: list[float], n_clusters=3):
        """エントロピー値をクラスタリングする"""
        try:
            from sklearn.cluster import KMeans
            
            # 1次元データをクラスタリングするために形状を変更
            X = np.array(entropies_list).reshape(-1, 1)
            
            # KMeansクラスタリングを実行
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            
            # 各クラスタの中心値
            cluster_centers = kmeans.cluster_centers_.flatten()
            
            # 各クラスタのサイズ（含まれるデータポイント数）
            cluster_sizes = [np.sum(cluster_labels == i) for i in range(n_clusters)]
            
            return {
                "labels": cluster_labels,
                "centers": cluster_centers,
                "sizes": cluster_sizes
            }
        except ImportError:
            # sklearnがない場合は簡易的なクラスタリング
            thresholds = np.linspace(0, 1, n_clusters+1)
            cluster_labels = np.zeros(len(entropies_list), dtype=int)
            
            for i in range(1, len(thresholds)):
                cluster_labels[(entropies_list >= thresholds[i-1]) & (entropies_list < thresholds[i])] = i-1
            
            cluster_sizes = [np.sum(cluster_labels == i) for i in range(n_clusters)]
            
            return {
                "labels": cluster_labels,
                "thresholds": thresholds,
                "sizes": cluster_sizes
            }


class PredictionCertaintyAnalyzer:
    def __init__(self, save_dir_path: Path, num_classes: int):
        """
        推論結果を解析するクラス。
        """
        self.save_dir_path = save_dir_path
        self.save_dir_path.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        self.num_scene_classes = 6

    def load_inference_results(self, csv_path: Path) -> InferenceResult:
        """
        CSVファイルから推論結果を読み込む。

        Args:
            csv_path: 読み込むCSVファイルのパス

        Returns:
            InferenceResult: 読み込んだ推論結果
        """
        try:
            image_paths, probabilities, labels = self._read_inference_results_csv(csv_path)
            return InferenceResult(image_paths=image_paths, probabilities=probabilities, labels=labels)
        except Exception as e:
            raise

    def _read_inference_results_csv(self, csv_path: Path):
        """CSVファイルを読み込み、データを分割するヘルパーメソッド
        
        Args:
            csv_path (Path): 読み込むCSVファイルのパス
            
        Returns:
            tuple:
                list[str]: 画像パスのリスト
                list[list[float]]: 確率値のリスト
                list[list[int]]: ラベルのリスト
        """
        image_paths = []
        probabilities = []
        ground_truth_labels = []

        with open(csv_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)  # ヘッダーをスキップ

            for row in reader:
                image_paths.append(row[0])
                # 確率値とラベルの区切り位置を計算
                num_probabilities = (len(row) - 1) // 2
                
                # 確率値とラベルを分割して追加
                probabilities.append(
                    list(map(float, row[1:num_probabilities + 1]))
                )
                ground_truth_labels.append(
                    list(map(int, row[num_probabilities + 1:]))
                )

        return image_paths, probabilities, ground_truth_labels

    def load_threshold_results(self, csv_path: Path) -> HardMultiLabelResult:
        """
        CSVファイルから閾値を適用した結果を読み込む。
        """
        try:
            image_paths, multi_labels, ground_truth_labels = self._read_threshold_results_csv(csv_path)
            return HardMultiLabelResult(image_paths=image_paths, multi_labels=multi_labels, ground_truth_labels=ground_truth_labels)
        except Exception as e:
            raise

    def _read_threshold_results_csv(self, csv_path: Path):
        """CSVファイルを読み込み、データを分割するヘルパーメソッド
        
        Args:
            csv_path (Path): 読み込むCSVファイルのパス
            
        Returns:
            tuple:
                list[str]: 画像パスのリスト
                list[list[float]]: 確率値のリスト
                list[list[int]]: ラベルのリスト
        """
        image_paths = []
        multi_labels = []
        ground_truth_labels = []

        with open(csv_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)  # ヘッダーをスキップ

            for row in reader:
                image_paths.append(row[0])
                # 確率値とラベルの区切り位置を計算
                num_multi_labels = (len(row) - 1) // 2
                
                # 確率値とラベルを分割して追加
                multi_labels.append(
                    list(map(float, row[1:num_multi_labels + 1]))
                )
                ground_truth_labels.append(
                    list(map(int, row[num_multi_labels + 1:]))
                )

        return image_paths, multi_labels, ground_truth_labels
    
    
    def analyze_prediction_certainty(self, inference_result: InferenceResult) -> list[float]:
        """
        予測確信度を解析する。
        """
        # 予測確信度を計算
        normalized_entropies = self._calculate_prediction_certainty(inference_result.probabilities)

        return normalized_entropies

    
    def _calculate_prediction_certainty(self, probabilities: list[list[float]]) -> list[float]:
        """
        予測確信度を計算するヘルパーメソッド
        
        Args:
            inference_result (InferenceResult): 推論結果
            
        Returns:
            float: 予測確信度
        """
        normalized_entropies = []
        # 推論結果（確率）の正規化エントロピーを計算
        for i in range(len(probabilities)):
            # 正規化エントロピー
            # すべてのクラスを使用して算出する場合
            # normalized_entropy = entropy(probabilities[i], base=2) / np.log2(self.num_classes)
            # シーンクラスのみを使用して算出する場合
            normalized_entropy = entropy([probabilities[i][j] for j in range(self.num_scene_classes)], base=2) / np.log2(self.num_scene_classes)
            normalized_entropies.append(normalized_entropy)

            # debug用
            # if i % 100 == 0:
            #     print(probabilities[i])
            #     print(normalized_entropy)

        return normalized_entropies
    

    def visualize_certainty(self, entropies_list: list[float], video_name: str):
        """
        予測確信度を可視化する。
        
        Args:
            entropies_list (list[float]): 予測確信度のリスト
            video_name (str): 動画名
        """
        n_images = len(entropies_list)
        
        # 時系列の画像を作成
        timeline_width = n_images
        timeline_height = n_images // 10
        
        # 保存パスの設定
        save_file = self.save_dir_path / f'{video_name}certainty_timeline.svg'
            
        # SVGドキュメントの作成
        dwg = svgwrite.Drawing(str(save_file), size=(timeline_width, timeline_height))
        dwg.add(dwg.rect((0, 0), (timeline_width, timeline_height), fill='white'))

        for i in range(n_images):
            x1 = i * (timeline_width // n_images)
            x2 = (i + 1) * (timeline_width // n_images)
            
            colored_certainty = int(255 * entropies_list[i])
            color = (colored_certainty, colored_certainty, colored_certainty)
            color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
            dwg.add(dwg.rect((x1, 0), (x2-x1, timeline_height), fill=color_hex))

        # SVGファイルを保存
        dwg.save()

    def analyze_prediction_accuracy_correlation(self, inference_result: InferenceResult) -> dict:
        """
        予測確率と正解ラベルの関係性を分析する

        Args:
            inference_result (InferenceResult): 推論結果

        Returns:
            dict: 分析結果
                - prob_label_diffs: 予測確率と正解ラベルの差の絶対値
                - normalized_entropies: 予測確率の正規化エントロピー
                - correlation_coefficient: スピアマンの順位相関係数
                - p_value: 相関係数の p値
        """
        from scipy import stats
        import numpy as np

        probabilities = inference_result.probabilities
        ground_truth_labels = inference_result.labels
        
        # 予測確率と正解ラベルの差の絶対値を計算
        prob_label_diffs = []
        for i in range(len(probabilities)):
            # シーンクラスのみを使用
            scene_probs = probabilities[i][:self.num_scene_classes]
            scene_labels = ground_truth_labels[i][:self.num_scene_classes]
            
            # 各クラスごとの差の平均を計算
            diff = np.mean([abs(prob - label) for prob, label in zip(scene_probs, scene_labels)])
            prob_label_diffs.append(diff)

        # 正規化エントロピーを計算
        normalized_entropies = self.analyze_prediction_certainty(inference_result)

        # スピアマンの順位相関係数を計算
        correlation_coefficient, p_value = stats.spearmanr(prob_label_diffs, normalized_entropies)

        return {
            'prob_label_diffs': prob_label_diffs,
            'normalized_entropies': normalized_entropies,
            'correlation_coefficient': correlation_coefficient,
            'p_value': p_value
        }

    def visualize_accuracy_entropy_relation(self, analysis_result: dict, inference_result: InferenceResult, video_name: str):
        """
        予測確率と正解ラベルの差、および確信度（エントロピー）の関係を散布図で可視化する
        予測の正解・不正解で色分けを行う

        Args:
            analysis_result (dict): analyze_prediction_accuracy_correlationの結果
            inference_result (InferenceResult): 推論結果
            video_name (str): 動画名
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 予測の正解・不正解を判定
        probabilities = inference_result.probabilities
        ground_truth_labels = inference_result.labels
        correct_predictions = []
        
        for i in range(len(probabilities)):
            # シーンクラスのみを使用
            scene_probs = probabilities[i][:self.num_scene_classes]
            scene_labels = ground_truth_labels[i][:self.num_scene_classes]
            
            # 閾値0.5で予測ラベルに変換
            pred_labels = [1 if prob >= 0.5 else 0 for prob in scene_probs]
            
            # 全クラスで正解したかどうか
            is_correct = all(p == g for p, g in zip(pred_labels, scene_labels))
            correct_predictions.append(is_correct)
        
        # 正解・不正解のインデックスを取得
        correct_idx = np.where(correct_predictions)[0]
        incorrect_idx = np.where(~np.array(correct_predictions))[0]
        
        plt.figure(figsize=(10, 6))
        
        # 正解のポイントを緑色で表示
        plt.scatter(np.array(analysis_result['normalized_entropies'])[correct_idx], 
                   np.array(analysis_result['prob_label_diffs'])[correct_idx], 
                   c='green', alpha=0.5, label='correct')
        
        # 不正解のポイントを赤色で表示
        plt.scatter(np.array(analysis_result['normalized_entropies'])[incorrect_idx], 
                   np.array(analysis_result['prob_label_diffs'])[incorrect_idx], 
                   c='red', alpha=0.5, label='incorrect')
        
        plt.xlabel('Normalized Entropy', fontsize=10)
        plt.ylabel('Absolute Difference between Prediction and Ground Truth', fontsize=10)
        plt.title(f'Relationship between Prediction Confidence and Accuracy\nCorrelation: {analysis_result["correlation_coefficient"]:.3f} (p-value: {analysis_result["p_value"]:.3e})', 
                fontsize=12)
        plt.legend()
        
        # 保存
        save_file = self.save_dir_path / f'{video_name}_accuracy_entropy_relation.png'
        plt.savefig(save_file)
        plt.close()

    def save_analysis_results_to_csv(self, analysis_result: dict, inference_result: InferenceResult, video_name: str):
        """
        分析結果をCSVファイルに保存する
        
        Args:
            analysis_result (dict): 分析結果
            inference_result (InferenceResult): 推論結果
            video_name (str): 動画名
        """
        import pandas as pd
        
        # 予測の正解・不正解を判定
        probabilities = inference_result.probabilities
        ground_truth_labels = inference_result.labels
        correct_predictions = []
        
        for i in range(len(probabilities)):
            scene_probs = probabilities[i][:self.num_scene_classes]
            scene_labels = ground_truth_labels[i][:self.num_scene_classes]
            pred_labels = [1 if prob >= 0.5 else 0 for prob in scene_probs]
            is_correct = all(p == g for p, g in zip(pred_labels, scene_labels))
            correct_predictions.append(is_correct)

        # データフレームの作成
        data = {
            'frame_number': range(len(probabilities)),
            'normalized_entropy': analysis_result['normalized_entropies'],
            'prediction_ground_truth_diff': analysis_result['prob_label_diffs'],
            'is_correct': correct_predictions,
            'ground_truth_labels': [','.join(map(str, labels[:self.num_scene_classes])) for labels in ground_truth_labels],
            'predicted_probabilities': [','.join(f'{prob:.3f}' for prob in probs[:self.num_scene_classes]) for probs in probabilities]
        }
        
        df = pd.DataFrame(data)
        
        # 相関係数などの全体の統計情報を追加
        stats_data = {
            'video_name': [video_name],
            'correlation_coefficient': [analysis_result['correlation_coefficient']],
            'p_value': [analysis_result['p_value']],
            'total_frames': [len(probabilities)],
            'correct_predictions': [sum(correct_predictions)],
            'accuracy': [sum(correct_predictions) / len(correct_predictions)]
        }
        stats_df = pd.DataFrame(stats_data)
        
        # CSVファイルとして保存
        frame_results_path = self.save_dir_path / f'{video_name}_frame_analysis.csv'
        stats_path = self.save_dir_path / f'{video_name}_statistics.csv'
        
        df.to_csv(frame_results_path, index=False)
        stats_df.to_csv(stats_path, index=False)

    def analyze_class_wise_accuracy(self, analysis_result: dict, inference_result: InferenceResult) -> dict:
        """
        各クラスごとの予測精度と確信度の関係を分析する
        
        Args:
            analysis_result (dict): analyze_prediction_accuracy_correlationの結果
            inference_result (InferenceResult): 推論結果
        
        Returns:
            dict: クラスごとの分析結果
        """
        probabilities = inference_result.probabilities
        ground_truth_labels = inference_result.labels
        class_wise_results = {}
        
        for class_idx in range(self.num_scene_classes):
            correct_predictions = []
            class_prob_diffs = []
            
            for i in range(len(probabilities)):
                # 各クラスの予測と正解を取得
                pred = 1 if probabilities[i][class_idx] >= 0.5 else 0
                true = ground_truth_labels[i][class_idx]
                
                # 正誤判定
                is_correct = (pred == true)
                correct_predictions.append(is_correct)
                
                # 予測確率と正解の差
                prob_diff = abs(probabilities[i][class_idx] - true)
                class_prob_diffs.append(prob_diff)
            
            # クラスごとの統計情報を計算
            from scipy import stats
            correlation_coefficient, p_value = stats.spearmanr(class_prob_diffs, analysis_result['normalized_entropies'])
            
            class_wise_results[f'class_{class_idx}'] = {
                'accuracy': sum(correct_predictions) / len(correct_predictions),
                'total_correct': sum(correct_predictions),
                'total_samples': len(correct_predictions),
                'correlation_coefficient': correlation_coefficient,
                'p_value': p_value,
                'prob_diffs': class_prob_diffs,
                'correct_predictions': correct_predictions
            }
        
        return class_wise_results

    def visualize_class_wise_accuracy(self, class_wise_results: dict, analysis_result: dict, video_name: str):
        """
        クラスごとの予測精度と確信度の関係を可視化する
        
        Args:
            class_wise_results (dict): analyze_class_wise_accuracyの結果
            analysis_result (dict): analyze_prediction_accuracy_correlationの結果
            video_name (str): 動画名
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
        
        for i, (class_name, results) in enumerate(class_wise_results.items()):
            correct_idx = np.where(results['correct_predictions'])[0]
            incorrect_idx = np.where(~np.array(results['correct_predictions']))[0]
            
            plt.subplot(2, 3, i+1)
            
            # 正解のポイントを表示
            plt.scatter(np.array(analysis_result['normalized_entropies'])[correct_idx],
                    np.array(results['prob_diffs'])[correct_idx],
                    c=colors[i], alpha=0.5, label='correct')
            
            # 不正解のポイントを表示
            plt.scatter(np.array(analysis_result['normalized_entropies'])[incorrect_idx],
                    np.array(results['prob_diffs'])[incorrect_idx],
                    c=colors[i], marker='x', alpha=0.5, label='incorrect')
            
            plt.xlabel('Normalized Entropy')
            plt.ylabel('Prediction Error')
            plt.title(f'{class_name}\nAcc: {results["accuracy"]:.3f}, Corr: {results["correlation_coefficient"]:.3f}')
            plt.legend()
        
        plt.tight_layout()
        save_file = self.save_dir_path / f'{video_name}_class_wise_analysis.png'
        plt.savefig(save_file)
        plt.close()

    def save_class_wise_results_to_csv(self, class_wise_results: dict, video_name: str):
        """
        クラスごとの分析結果をCSVファイルに保存する
        
        Args:
            class_wise_results (dict): クラスごとの分析結果
            video_name (str): 動画名
        """
        import pandas as pd
        
        # クラスごとの統計情報
        stats_data = {
            'class': [],
            'accuracy': [],
            'total_correct': [],
            'total_samples': [],
            'correlation_coefficient': [],
            'p_value': []
        }
        
        for class_name, results in class_wise_results.items():
            stats_data['class'].append(class_name)
            stats_data['accuracy'].append(results['accuracy'])
            stats_data['total_correct'].append(results['total_correct'])
            stats_data['total_samples'].append(results['total_samples'])
            stats_data['correlation_coefficient'].append(results['correlation_coefficient'])
            stats_data['p_value'].append(results['p_value'])
        
        stats_df = pd.DataFrame(stats_data)
        stats_path = self.save_dir_path / f'{video_name}_class_wise_statistics.csv'
        stats_df.to_csv(stats_path, index=False)

    def aggregate_all_results(self, all_class_wise_results: dict[str, dict], save_dir: Path):
        """
        すべての動画のクラスごとの分析結果を集計する
        
        Args:
            all_class_wise_results (dict): 動画名をキーとし、その動画のクラスごとの分析結果を値とする辞書
            save_dir (Path): 保存先ディレクトリ
        """
        import pandas as pd
        
        # 全動画の統計情報を格納するリスト
        aggregated_stats = {
            'video_name': [],
            'class': [],
            'accuracy': [],
            'total_correct': [],
            'total_samples': [],
            'correlation_coefficient': [],
            'p_value': []
        }
        
        # クラスごとの統計量を計算するための一時保存用
        class_wise_aggregation = {}
        
        # 各動画の結果を集計
        for video_name, class_results in all_class_wise_results.items():
            for class_name, results in class_results.items():
                # 個々の動画の結果を保存
                aggregated_stats['video_name'].append(video_name)
                aggregated_stats['class'].append(class_name)
                aggregated_stats['accuracy'].append(results['accuracy'])
                aggregated_stats['total_correct'].append(results['total_correct'])
                aggregated_stats['total_samples'].append(results['total_samples'])
                aggregated_stats['correlation_coefficient'].append(results['correlation_coefficient'])
                aggregated_stats['p_value'].append(results['p_value'])
                
                # クラスごとの集計用データを蓄積
                if class_name not in class_wise_aggregation:
                    class_wise_aggregation[class_name] = {
                        'accuracies': [],
                        'correlation_coefficients': [],
                        'total_correct': 0,
                        'total_samples': 0
                    }
                
                class_wise_aggregation[class_name]['accuracies'].append(results['accuracy'])
                class_wise_aggregation[class_name]['correlation_coefficients'].append(results['correlation_coefficient'])
                class_wise_aggregation[class_name]['total_correct'] += results['total_correct']
                class_wise_aggregation[class_name]['total_samples'] += results['total_samples']
        
        # 詳細な結果をCSVに保存
        detailed_df = pd.DataFrame(aggregated_stats)
        detailed_df.to_csv(save_dir / 'all_videos_detailed_results.csv', index=False)
        
        # クラスごとの集計結果を計算
        summary_stats = {
            'class': [],
            'mean_accuracy': [],
            'std_accuracy': [],
            'mean_correlation': [],
            'std_correlation': [],
            'total_accuracy': [],  # 全サンプルでの正解率
            'total_samples': []
        }
        
        for class_name, stats in class_wise_aggregation.items():
            summary_stats['class'].append(class_name)
            summary_stats['mean_accuracy'].append(np.mean(stats['accuracies']))
            summary_stats['std_accuracy'].append(np.std(stats['accuracies']))
            summary_stats['mean_correlation'].append(np.mean(stats['correlation_coefficients']))
            summary_stats['std_correlation'].append(np.std(stats['correlation_coefficients']))
            summary_stats['total_accuracy'].append(stats['total_correct'] / stats['total_samples'])
            summary_stats['total_samples'].append(stats['total_samples'])
        
        # 集計結果をCSVに保存
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(save_dir / 'all_videos_summary_statistics.csv', index=False)
    
    def visualize_aggregated_class_wise_plots(self, all_class_wise_results: dict[str, dict], 
                                        all_analysis_results: dict[str, dict], save_dir: Path):
        """
        全動画のデータをまとめたクラスごとの散布図を作成する

        Args:
            all_class_wise_results (dict): 動画名をキーとし、クラスごとの分析結果を値とする辞書
            all_analysis_results (dict): 動画名をキーとし、分析結果を値とする辞書
            save_dir (Path): 保存先ディレクトリ
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as stats
        
        plt.figure(figsize=(15, 10))
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
        
        # クラスごとのデータを集約
        aggregated_data = {f'class_{i}': {'entropies': [], 'prob_diffs': [], 'is_correct': []} 
                            for i in range(self.num_scene_classes)}
        
        # 全動画のデータを集約
        for video_name, class_results in all_class_wise_results.items():
            analysis_result = all_analysis_results[video_name]
            
            for class_name, results in class_results.items():
                aggregated_data[class_name]['entropies'].extend(analysis_result['normalized_entropies'])
                aggregated_data[class_name]['prob_diffs'].extend(results['prob_diffs'])
                aggregated_data[class_name]['is_correct'].extend(results['correct_predictions'])
        
        # クラスごとの散布図を作成
        for i, (class_name, data) in enumerate(aggregated_data.items()):
            plt.subplot(2, 3, i+1)
            
            # データをnumpy配列に変換
            entropies = np.array(data['entropies'])
            prob_diffs = np.array(data['prob_diffs'])
            is_correct = np.array(data['is_correct'])
            
            # 正解・不正解のインデックスを取得
            correct_idx = np.where(is_correct)[0]
            incorrect_idx = np.where(~is_correct)[0]
            
            # 正解のポイントを表示（丸い点）
            plt.scatter(entropies[correct_idx], prob_diffs[correct_idx],
                        c=colors[i], alpha=0.7, label='correct', s=20, marker='o')
            
            # 不正解のポイントを表示（×の点）
            plt.scatter(entropies[incorrect_idx], prob_diffs[incorrect_idx],
                        c=colors[i], marker='x', alpha=0.5, label='incorrect', s=40)
            
            # 相関係数を計算
            correlation_coefficient, p_value = stats.spearmanr(prob_diffs, entropies)
            accuracy = len(correct_idx) / len(is_correct)
            
            plt.xlabel('Normalized Entropy')
            plt.ylabel('Prediction Error')
            plt.title(f'{class_name}\nAcc: {accuracy:.3f}, Corr: {correlation_coefficient:.3f}')
            plt.legend()
        
        plt.tight_layout()
        save_file = save_dir / 'all_videos_class_wise_analysis.png'
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()