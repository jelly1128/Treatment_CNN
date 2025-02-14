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
    
    test_dataloaders = dataloader_factory.create_multilabel_test_dataloaders(test_data_dirs)
    
    # visualize
    # テストデータの最初のデータを表示
    # show_dataset_stats(test_dataloader[test_data_dirs[0]])
    
    model = setup_model(config, device, num_gpus, mode='test')
    
    # 推論
    inference = Inference(model, device)
    results = inference.run(config.paths.save_dir, test_dataloaders)
    
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