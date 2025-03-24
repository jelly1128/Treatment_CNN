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