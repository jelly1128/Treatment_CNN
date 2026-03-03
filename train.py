import argparse
from pathlib import Path

from config.config_loader import load_experiment_config
from engine.runner import CVRunner


def parse_args():
    """コマンドライン引数を解析する。

    Returns:
        argparse.Namespace: 解析済み引数（config, fold）。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, help='Path to config file', default='config.yaml')
    parser.add_argument('-f', '--fold', type=int, help='Fold number to train (optional, if not set, trains all folds)')
    return parser.parse_args()


def main():
    """設定ファイルを読み込み、CVRunnerで学習を実行する。"""
    # 設定読み込み
    args = parse_args()
    config = load_experiment_config(args.config)

    if config.mode != 'train':
        raise ValueError(f"config.yamlのmodeが 'train' 以外に設定されています (現在の設定: {config.mode})")

    # Runner
    runner = CVRunner(config)

    if args.fold is not None:
        # 指定されたfoldのみ学習
        print(f"=== Fold {args.fold} の学習を開始します ===")
        runner.run_single_fold_train(args.fold)
        print(f"=== Fold {args.fold} の学習が完了しました ===")
    else:
        # 全 fold 実行
        n_folds = len(runner.splitter)
        print(f"=== {n_folds} folds の学習を開始します ===")
        results = runner.run_all_folds_train()
        print(f"=== {len(results.fold_results)}/{n_folds} folds の学習が完了しました ===")

        if len(results.fold_results) < n_folds:
            failed = n_folds - len(results.fold_results)
            print(f"警告: {failed} fold(s) が失敗しました。ログを確認してください。")


if __name__ == '__main__':
    main()
