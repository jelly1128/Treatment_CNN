import argparse
import logging
from pathlib import Path

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
from evaluate.save_metrics import save_metrics_to_csv
def test(config: dict, test_data_dirs: list):
    # setup
    device, num_gpus = get_device_and_num_gpus()
    set_seed(42)
    
    dataloader_factory = DataLoaderFactory(
        dataset_root=config.paths.dataset_root,
        batch_size=1,
        num_classes=config.test.num_classes,
        num_gpus=num_gpus
    )
    """
    test_dataloaders = dataloader_factory.create_multilabel_test_dataloaders(test_data_dirs)
    
    # visualize
    # テストデータの最初のデータを表示
    # show_dataset_stats(test_dataloader[test_data_dirs[0]])
    
    model = setup_model(config, device, num_gpus, mode='test')
    
    # 推論
    inference = Inference(model, device)
    results = inference.run(config.paths.save_dir, test_dataloaders)
    """
    
    # 可視化
    visualizer = ResultsVisualizer(config.paths.save_dir)
    results = {}
    # debug
    for folder_name in ["20210524100043_000001-001", "20210531112330_000001-001"]:
        results[folder_name] = visualizer.load_results(Path(config.paths.save_dir) / folder_name / f'raw_results_{folder_name}.csv')
    
    # コンバーター
    converter = MultiToSingleLabelConverter(results)
    
    # マルチラベルを閾値でハードラベルに変換した結果
    hard_multilabel_results = converter.convert_soft_to_hard_multilabels(threshold=0.5)
    visualizer.save_multilabel_visualization(hard_multilabel_results)
    
    # 混同行列の計算
    calculator = ClassificationMetricsCalculator()
    video_metrics = calculator.calculate_metrics_per_video(hard_multilabel_results)
    overall_metrics = calculator.calculate_overall_metrics(hard_multilabel_results)
    save_metrics_to_csv(video_metrics, overall_metrics, Path(config.paths.save_dir) / 'video_metrics.csv', Path(config.paths.save_dir) / 'overall_metrics.csv')
    
    # 出力を解析
    # analyzer = Analyzer(config.paths.save_dir, config.test.num_classes)
    # analyzer.analyze(results)
    
    # tester = Tester(config, device, num_gpus, test_data_dirs)
    # tester.test()

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-class organ classification model using ResNet')
    parser.add_argument('-c', '--config', help='Path to config file', default='config.yaml')
    return parser.parse_args()
    
def main():
    args = parse_args()
    config = load_test_config(args.config)
    
    # 結果保存フォルダを作成
    Path(config.paths.save_dir).mkdir(exist_ok=True)
    
    setup_logging(config.paths.save_dir, mode='test')
    
    # dataloaderの作成
    splitter = CrossValidationSplitter(splits=config.splits.root)
    split_folders = splitter.get_split_folders()
    
    # debug
    # fold_idx=0のtestのデータディレクトリを取得
    test_data_dirs = split_folders[0]['test']
    print(test_data_dirs)
    
    test(config, test_data_dirs)

if __name__ == '__main__':
    main()