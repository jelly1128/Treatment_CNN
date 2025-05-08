import argparse
import logging
from pathlib import Path

from config.config_loader import load_test_config
from data.data_splitter import CrossValidationSplitter
from data.dataloader import DataLoaderFactory
from engine.inference import Inference
from evaluate.analyzer import Analyzer
from utils.torch_utils import get_device_and_num_gpus, set_seed
from utils.logger import setup_logging
from utils.window_key import WindowSizeKey
from evaluate.results_visualizer import ResultsVisualizer
from labeling.label_converter import MultiToSingleLabelConverter
from evaluate.metrics import ClassificationMetricsCalculator
from evaluate.save_metrics import save_video_metrics_to_csv, save_overall_metrics_to_csv
from model.setup_models import setup_model


def test_single_label(config: dict, 
         test_data_dirs: list, 
         model_path: Path, 
         save_dir_path: Path, 
         window_sizes: list, 
         logger: logging.Logger):
    """
    シングルラベル用のテストデータ評価関数
    """
    device, num_gpus = get_device_and_num_gpus()
    set_seed(42)

    # シングルラベル用データローダー
    dataloader_factory = DataLoaderFactory(
        dataset_root=config.paths.dataset_root,
        batch_size=1,
        num_classes=config.test.num_classes,
        num_gpus=num_gpus
    )
    merge_label_indices = [4, 5, 6, 11, 12]
    merge_to_label = 4
    test_dataloaders = dataloader_factory.create_single_label_test_dataloaders(
        test_data_dirs, merge_label_indices, merge_to_label=merge_to_label
    )

    # モデルのセットアップ
    model = setup_model(
        config=config, 
        device=device, 
        num_gpus=num_gpus, 
        mode='test', 
        model_path=model_path
    )

    # 推論
    inference = Inference(model, device)
    results = inference.run(save_dir_path, test_dataloaders, mode='single_label')

    # 可視化
    visualizer = ResultsVisualizer(save_dir_path)
    visualizer.save_single_label_visualization(results)

    # メトリクス計算
    calculator = ClassificationMetricsCalculator(num_classes=config.test.num_classes, mode="single_label")
    video_metrics = calculator.calculate_single_label_metrics_per_video(results)
    overall_metrics = calculator.calculate_single_label_overall_metrics(results)

    # メトリクス保存
    save_video_metrics_to_csv(video_metrics, save_dir_path, methods='single_label')
    save_overall_metrics_to_csv(overall_metrics, save_dir_path / 'fold_results', methods='single_label')
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Single-label classification test')
    parser.add_argument('-c', '--config', help='Path to config file', default='config.yaml')
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_test_config(args.config)
    Path(config.paths.save_dir).mkdir(exist_ok=True)
    splitter = CrossValidationSplitter(splits=config.splits.root)
    split_folders = splitter.get_split_folders()
    window_sizes = [1, 11]
    all_folds_single_label_results = {}

    for fold_idx, (split_data, model_path) in enumerate(zip(split_folders, config.paths.model_paths)):
        fold_save_dir_path = Path(config.paths.save_dir) / f"fold_{fold_idx}"
        fold_save_dir_path.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(fold_save_dir_path, f'test_fold_{fold_idx}')
        logger.info(f"Started test for fold {fold_idx}")
        results = test_single_label(
            config=config, 
            test_data_dirs=split_data['test'],
            model_path=model_path,
            save_dir_path=fold_save_dir_path,
            window_sizes=window_sizes,
            logger=logger
        )
        # 各foldの結果を集約
        for folder_name, result in results.items():
            all_folds_single_label_results[folder_name] = result

    # 全foldの集約結果で全体メトリクスを計算・保存
    calculator = ClassificationMetricsCalculator(num_classes=config.test.num_classes, mode="single_label")
    metrics = calculator.calculate_all_folds_metrics(all_folds_single_label_results, Path(config.paths.save_dir))

if __name__ == '__main__':
    main()
