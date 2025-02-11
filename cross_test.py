import os
import argparse

from config.config import load_test_config
from engine.tester import Tester
from utils.torch_utils import get_device_and_num_gpus, set_seed
from utils.logger import setup_logging


SPLIT1 = (
    "20210119093456_000001-001",
    "20210531112330_000005-001",
    "20211223090943_000001-002",
    "20230718-102254-ES06_20230718-102749-es06-hd",
    "20230802-104559-ES09_20230802-105630-es09-hd",
)

SPLIT2 = (
    "20210119093456_000001-002",
    "20210629091641_000001-002",
    "20211223090943_000001-003",
    "20230801-125025-ES06_20230801-125615-es06-hd",
    "20230803-110626-ES06_20230803-111315-es06-hd"
)

SPLIT3 = (
    "20210119093456_000002-001",
    "20210630102301_000001-002",
    "20220322102354_000001-002",
    "20230802-095553-ES09_20230802-101030-es09-hd",
    "20230803-093923-ES09_20230803-094927-es09-hd",
)

SPLIT4 = (
    "20210524100043_000001-001",
    "20210531112330_000001-001",
    "20211021093634_000001-001",
    "20211021093634_000001-003"
)

LABELING_SPLIT = (
    # "20230801-125025-ES06_20230801-125615-es06-hd",
    # "20230802-095553-ES09_20230802-101030-es09-hd",
    "20210119093456_000002-001",
    # "20210524100043_000001-001",
    # "20211223090943_000001-003",
    # "20220322102354_000001-002",
)

TEST_SPLIT = SPLIT4


def test(config):
    # setup
    device, num_gpus = get_device_and_num_gpus()
    set_seed(42)
    
    tester = Tester(config, device, num_gpus, TEST_SPLIT)
    
    tester.test()
    
    os._exit(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-class organ classification model using ResNet')
    parser.add_argument('-c', '--config', help='Path to config file', default='config.yaml')
    return parser.parse_args()
    
def main():
    args = parse_args()
    config = load_test_config(args.config)
    
    # 結果保存フォルダを作成
    os.makedirs(config.paths.save_dir, exist_ok=True)
    
    setup_logging(config.paths.save_dir, mode='test')
    
    test(config)

if __name__ == '__main__':
    main()