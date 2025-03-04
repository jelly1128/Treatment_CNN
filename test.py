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
from evaluate.save_metrics import save_video_metrics_to_csv, save_overall_metrics_to_csv
from analyze_0218 import load_hard_multilabel_results


def test(config: dict, test_data_dirs: list, model_path: str, save_dir: str, window_sizes: list):
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
    # コンバーター
    converter = MultiToSingleLabelConverter(results)
    
    # マルチラベルを閾値でハードラベルに変換する手法の結果
    ## マルチラベルを閾値でハードラベルに変換
    hard_multilabel_results = converter.convert_soft_to_hard_multilabels(threshold=0.5)
    converter.save_hard_multilabel_results(hard_multilabel_results, Path(save_dir), methods = 'threshold_50%')

    # 正解ラベルの可視化
    visualizer.save_main_classes_visualization(hard_multilabel_results)
    # マルチラベルを閾値でハードラベルに変換した結果を可視化
    visualizer.save_multilabel_visualization(hard_multilabel_results, Path(save_dir), methods = 'threshold_50%')


    ## 混同行列の計算
    calculator = ClassificationMetricsCalculator(num_classes=6)
    video_metrics = calculator.calculate_multilabel_metrics_per_video(hard_multilabel_results)
    overall_metrics = calculator.calculate_multilabel_overall_metrics(hard_multilabel_results)
    ## 各動画フォルダにマルチラベルのメトリクスを保存
    save_video_metrics_to_csv(video_metrics, Path(save_dir), methods = 'threshold_50%')
    ## 全体のメトリクスを保存
    save_overall_metrics_to_csv(overall_metrics, Path(save_dir) / 'fold_results', methods = 'threshold_50%')
    
    # スライディングウィンドウ解析
    analyzer = Analyzer(save_dir, config.test.num_classes)

    all_window_results = analyzer.analyze_sliding_windows(
        Path(save_dir),
        hard_multilabel_results,
        visualizer,
        calculator,
        window_sizes=window_sizes
    )
    
    return hard_multilabel_results, all_window_results
    

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
        
    # 全foldの結果を保存する辞書
    all_folds_hard_multilabel_results = {}
    all_folds_all_window_results = {}
    
    # window_sizeの探索範囲
    # window_sizes = list(range(9, 13, 2))
    window_sizes = [1, 11]
    # window_sizeごとに結果を保存する辞書を作成
    for window_size in window_sizes:
        if window_size not in all_folds_all_window_results:
            all_folds_all_window_results[f'window_size_{window_size}'] = {}
    
    
    for fold_idx, (split_data, model_path) in enumerate(zip(split_folders, config.paths.model_paths)):
        # fold用の結果保存フォルダを作成
        fold_save_dir = Path(config.paths.save_dir) / f"fold_{fold_idx}"
        fold_save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Started test for fold {fold_idx}")

        hard_multilabel_results, all_window_results = test(config=config, 
                                                           test_data_dirs=split_data['test'],
                                                           model_path=model_path,
                                                           save_dir=fold_save_dir,
                                                           window_sizes=window_sizes
                                                           )
        
        # 各foldの結果を全体の辞書に追加
        for folder_name, result in hard_multilabel_results.items():
            if folder_name not in all_folds_hard_multilabel_results:
                all_folds_hard_multilabel_results[folder_name] = result

        # Then store the results
        for window_size, sliding_window_results in all_window_results.items():
            for folder_name, result in sliding_window_results.items():
                all_folds_all_window_results[window_size][folder_name] = result

    # 全foldの結果を集約して評価指標を計算
    calculator = ClassificationMetricsCalculator(num_classes=6)

    # 各window_sizeの全foldの結果を集約して評価指標を計算
    for window_size, all_window_results in all_folds_all_window_results.items():
        save_dir = Path(config.paths.save_dir) / f'{window_size}'
        save_dir.mkdir(parents=True, exist_ok=True)
        metrics = calculator.calculate_all_folds_metrics(all_window_results, save_dir)



# def main():
#     args = parse_args()
#     config = load_test_config(args.config)
    
#     # 結果保存フォルダを作成
#     Path(config.paths.save_dir).mkdir(exist_ok=True)
    
#     # dataloaderの作成
#     splitter = CrossValidationSplitter(splits=config.splits.root)
#     split_folders = splitter.get_split_folders()
    
#     # window_sizeのリスト
#     window_sizes = range(1, 7, 2)
#     results_by_window = {}
    
#     for window_size in window_sizes:
#         logging.info(f"Processing window size: {window_size}")
        
#         # 全foldの結果を保存する辞書
#         all_folds_results = {}
        
#         # 集約した結果を保存するディレクトリ
#         # save_path = Path(f"/home/tanaka/0218/Treatment_CNN/{config.test.num_classes}class_resnet18_test") / f'all_folds_results_window_{window_size}'
#         save_path = Path(f"{config.paths.save_dir}") / f'all_folds_results_window_{window_size}'
#         save_path.mkdir(exist_ok=True)
        
#         for fold_idx, (split_data, model_path) in enumerate(zip(split_folders, config.paths.model_paths)):
#             # fold用の結果保存フォルダを作成
#             fold_save_dir = Path(config.paths.save_dir) / f"fold_{fold_idx}"
#             fold_save_dir.mkdir(parents=True, exist_ok=True)

#             print(f"Started test for fold {fold_idx}")

#             sliding_window_results = test(config=config, 
#                                           test_data_dirs=split_data['test'], 
#                                           model_path=model_path,
#                                           save_dir=fold_save_dir,
#                                           window_size=window_size
#                                           )
            
#             # 各foldの結果を全体の辞書に追加
#             for folder_name, result in sliding_window_results.items():
#                 if folder_name not in all_folds_results:
#                     all_folds_results[folder_name] = result
        

if __name__ == '__main__':
    main()