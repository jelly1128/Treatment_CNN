import argparse
from pathlib import Path

from config.config_loader import load_experiment_config
from engine.runner import CVRunner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, help='Path to config file', default='config.yaml')
    parser.add_argument('-f', '--fold', type=int, help='Fold number to test (optional, if not set, tests all folds)')
    return parser.parse_args()


def main():
    # 設定読み込み
    args = parse_args()
    config = load_experiment_config(args.config)

    if config.mode != 'test':
        raise ValueError(f"config.yamlのmodeが 'test' 以外に設定されています (現在の設定: {config.mode})")

    # Runner
    runner = CVRunner(config)

    if args.fold is not None:
        # 指定されたfoldのみテスト
        print(f"=== Fold {args.fold} のテストを開始します ===")
        runner.run_single_fold_test(args.fold)
        print(f"=== Fold {args.fold} のテストが完了しました ===")
    else:
        # 全 fold 実行
        n_folds = len(runner.splitter)
        print(f"=== {n_folds} folds のテストを開始します ===")
        results = runner.run_all_folds_test()
        print(f"=== {len(results.fold_results)}/{n_folds} folds のテストが完了しました ===")

        if len(results.fold_results) < n_folds:
            failed = n_folds - len(results.fold_results)
            print(f"警告: {failed} fold(s) が失敗しました。ログを確認してください。")


if __name__ == '__main__':
    main()
