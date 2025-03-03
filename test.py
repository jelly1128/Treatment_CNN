import argparse
import logging
from pathlib import Path
import csv
import pandas as pd

from config.config_loader import load_test_config
from data.data_splitter import CrossValidationSplitter
from data.dataloader import DataLoaderFactory
from data.dataset_visualizer import plot_dataset_samples, show_dataset_stats
from engine.inference import Inference
from evaluate.analyzer import Analyzer
from model.setup_models import setup_model
from utils.torch_utils import get_device_and_num_gpus, set_seed
from utils.logger import setup_logging
from evaluate.results_visualizer import ResultsVisualizer
from engine.inference import InferenceResult
from labeling.label_converter import MultiToSingleLabelConverter
from evaluate.metrics import ClassificationMetricsCalculator
# from evaluate.save_metrics import MetricsSaver
from evaluate.save_metrics import save_video_metrics_to_csv, save_overall_metrics_to_csv
from analyze_0218 import load_hard_multilabel_results


def test(config: dict, test_data_dirs: list, model_path: str, save_dir: str, window_size: int):
    """
    テストデータを用いてモデルの評価を行う関数
    Args:
        config (dict): 設定情報
        test_data_dirs (list): テストデータのディレクトリのリスト
        model_path (str): モデルのパス
        save_dir (str): 結果を保存するディレクトリ
        window_size (int): スライディングウィンドウのサイズ
    """
    # setup
    device, num_gpus = get_device_and_num_gpus()
    set_seed(42)
    
    dataloader_factory = DataLoaderFactory(
        dataset_root=config.paths.dataset_root,
        batch_size=1,
        num_classes=config.test.num_classes,
        num_gpus=num_gpus
    )
    test_dataloaders = dataloader_factory.create_multilabel_test_dataloaders(test_data_dirs)
    
    model = setup_model(config, device, num_gpus, mode='test', model_path=model_path)
    
    # 推論
    inference = Inference(model, device)
    results = inference.run(save_dir, test_dataloaders)
    
    
    # 可視化
    visualizer = ResultsVisualizer(save_dir)
    # results = {}
    # # debug
    # for folder_name in test_data_dirs:
    #     results[folder_name] = visualizer.load_results(Path(save_dir) / folder_name / f'raw_results_{folder_name}.csv')
    
    # コンバーター
    converter = MultiToSingleLabelConverter(results)
    # メトリクス保存
    # metrics_saver = MetricsSaver(Path(save_dir))
    
    # マルチラベルを閾値でハードラベルに変換する手法の結果
    ## マルチラベルを閾値でハードラベルに変換
    hard_multilabel_results = converter.convert_soft_to_hard_multilabels(threshold=0.5)
    converter.save_hard_multilabel_results(hard_multilabel_results, Path(save_dir), methods = 'threshold_50%')
    # 正解ラベルの可視化
    visualizer.save_main_classes_visualization(hard_multilabel_results)

    
    import sys
    sys.exit()
    
    ## マルチラベルを閾値でハードラベルに変換した結果を可視化
    visualizer.save_multilabel_visualization(hard_multilabel_results, methods = 'threshold_50%')
    ## 混同行列の計算
    calculator = ClassificationMetricsCalculator()
    video_metrics = calculator.calculate_multilabel_metrics_per_video(hard_multilabel_results)
    overall_metrics = calculator.calculate_multilabel_overall_metrics(hard_multilabel_results)
    # ## 各動画フォルダにマルチラベルのメトリクスを保存
    save_video_metrics_to_csv(video_metrics, Path(save_dir), methods = 'threshold_50%')
    save_overall_metrics_to_csv(overall_metrics, Path(save_dir), methods = 'threshold_50%')
    
    # ハードラベルの結果を読み込む
    hard_multilabel_results = {}
    for folder_name in test_data_dirs:
        csv_path = Path(f'{save_dir}/{folder_name}/threshold_50%/threshold_50%_results_{folder_name}.csv')
        hard_multilabel_results[folder_name] = load_hard_multilabel_results(csv_path)
        
    # print(hard_multilabel_results.keys())
    # print(len(hard_multilabel_results['20210524100043_000001-001'].multilabels))
    # print(len(hard_multilabel_results['20210524100043_000001-001'].ground_truth_labels))
    
    
    # スライディングウィンドウ解析
    # analyzer = Analyzer(save_dir, config.test.num_classes)
    # all_window_results, all_window_metrics = analyzer.analyze_sliding_windows(
    #     hard_multilabel_results,
    #     visualizer,
    #     calculator
    # )
    
    # スライディングウィンドウを適用して、平滑化されたラベルを生成
    analyzer = Analyzer(save_dir, config.test.num_classes)
    
    # スライディングウィンドウの結果を保存するディレクトリを作成
    sliding_window_dir = Path(save_dir) / f'sliding_window_{window_size}_results'
    sliding_window_dir.mkdir(exist_ok=True)
    
    # スライディングウィンドウの適用
    sliding_window_results = analyzer.apply_sliding_window_to_hard_multilabel_results(
        hard_multilabel_results, 
        window_size=window_size
    )
    
    # 平滑化されたラベルの可視化を新しいディレクトリに保存
    visualizer.save_singlelabel_visualization(
        sliding_window_results, 
        save_path=sliding_window_dir,
        methods=f'sliding_window_{window_size}'
    )
    
    # 平滑化されたラベルのメトリクスを計算
    sliding_window_video_metrics = calculator.calculate_singlelabel_metrics_per_video(sliding_window_results)
    sliding_window_overall_metrics = calculator.calculate_singlelabel_overall_metrics(sliding_window_results)
    
    # 各動画フォルダのメトリクスを新しいディレクトリに保存
    save_video_metrics_to_csv(
        sliding_window_video_metrics, 
        sliding_window_dir, 
        methods=f'sliding_window_{window_size}'
    )
    
    # 全体のメトリクスを新しいディレクトリに保存
    save_overall_metrics_to_csv(
        sliding_window_overall_metrics, 
        sliding_window_dir, 
        methods=f'sliding_window_{window_size}'
    )
    
    return sliding_window_results
    

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-class organ classification model using ResNet')
    parser.add_argument('-c', '--config', help='Path to config file', default='config.yaml')
    return parser.parse_args()
    
def main():
    args = parse_args()
    config = load_test_config(args.config)
    
    # 結果保存フォルダを作成
    Path(config.paths.save_dir).mkdir(exist_ok=True)
    
    # dataloaderの作成
    splitter = CrossValidationSplitter(splits=config.splits.root)
    split_folders = splitter.get_split_folders()
    
    # window_sizeのリスト
    window_sizes = range(1, 7, 2)
    results_by_window = {}
    
    for window_size in window_sizes:
        logging.info(f"Processing window size: {window_size}")
        
        # 全foldの結果を保存する辞書
        all_folds_results = {}
        
        # 集約した結果を保存するディレクトリ
        # save_path = Path(f"/home/tanaka/0218/Treatment_CNN/{config.test.num_classes}class_resnet18_test") / f'all_folds_results_window_{window_size}'
        save_path = Path(f"{config.paths.save_dir}") / f'all_folds_results_window_{window_size}'
        save_path.mkdir(exist_ok=True)
        
        for fold_idx, (split_data, model_path) in enumerate(zip(split_folders, config.paths.model_paths)):
            # fold用の結果保存フォルダを作成
            fold_save_dir = Path(config.paths.save_dir) / f"fold_{fold_idx}"
            fold_save_dir.mkdir(parents=True, exist_ok=True)

            print(f"Started test for fold {fold_idx}")

            sliding_window_results = test(config=config, 
                                          test_data_dirs=split_data['test'], 
                                          model_path=model_path,
                                          save_dir=fold_save_dir,
                                          window_size=window_size
                                          )
            
            # 各foldの結果を全体の辞書に追加
            for folder_name, result in sliding_window_results.items():
                if folder_name not in all_folds_results:
                    all_folds_results[folder_name] = result
        
        # 全foldの結果を集約して評価指標を計算
        # calculator = ClassificationMetricsCalculator()
        # metrics = calculator.calculate_all_folds_metrics(all_folds_results, save_path)
        
        # window_sizeごとの結果を保存
        # results_by_window[window_size] = metrics
    
    # window_size比較結果の保存
    # comparison_save_dir = Path(f"/home/tanaka/0218/Treatment_CNN/{config.test.num_classes}class_resnet18_test") / 'window_size_comparison'
    # comparison_save_dir.mkdir(exist_ok=True)
    # create_window_size_comparison(results_by_window, comparison_save_dir)

def create_window_size_comparison(results_by_window: dict, save_dir: Path):
    """
    全window_sizeのクラス別指標を比較するCSVを生成
    """
    window_summaries = []
    
    for window_size, results in results_by_window.items():
        # 各クラスの指標を抽出
        for class_idx in range(6):
            metrics = results['class_metrics'][class_idx]
            
            summary = {
                'window_size': window_size,
                'class': class_idx,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'accuracy': metrics['accuracy']
            }
            window_summaries.append(summary)
    
    # DataFrameに変換してソート
    df = pd.DataFrame(window_summaries)
    df = df.sort_values(['class', 'window_size'])
    
    # CSV保存
    comparison_path = save_dir / 'window_size_class_metrics.csv'
    df.to_csv(comparison_path, index=False)
    logging.info(f'Window size comparison saved: {comparison_path}')

if __name__ == '__main__':
    main()