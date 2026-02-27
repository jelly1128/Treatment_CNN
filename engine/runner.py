from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.optim as optim

from config.schema import ExperimentConfig
from data.data_splitter import CVSplitter, FoldSplit
from data.dataloader import DataLoaderFactory
from engine.trainer import Trainer
from engine.validator import Validator
from engine.inference import Inference
from model.cnn_models import SingleLabelClassificationModel, MultiTaskClassificationModel
from utils.torch_utils import get_device_and_num_gpus, set_seed
from utils.logger import logging, setup_logging
from utils.training_monitor import TrainingMonitor
from evaluate.analyzer import Analyzer
from evaluate.results_visualizer import ResultsVisualizer
from labeling.label_converter import MultiToSingleLabelConverter
from evaluate.metrics import ClassificationMetricsCalculator
from evaluate.save_metrics import save_video_metrics_to_csv, save_overall_metrics_to_csv
from utils.window_key import WindowSizeKey


@dataclass
class FoldResult:
    """
    1つの fold の実行結果。
    
    Attributes:
        fold_idx: fold のインデックス
        metrics: 評価指標（test モード時のみ）
        loss_history: 学習履歴（train モード時のみ）
    """
    fold_idx: int
    metrics: dict | None = None
    loss_history: dict | None = None


@dataclass
class AggregatedResult:
    """
    全 fold の集約結果。
    
    Attributes:
        fold_results: 各 fold の結果
        overall_metrics: 全 fold の平均・標準偏差など
    """
    fold_results: dict[int, FoldResult]
    overall_metrics: dict | None = None


class CVRunner:
    """
    交差検証(CrossValidation)の全体フローを管理するクラス。
    train / test 両モードに対応し、全 fold / 単一 fold の実行を選択できる。
    
    使用例:
        config = load_experiment_config('config.yaml')
        runner = CVRunner(config)
        
        # 全 fold 実行
        results = runner.run_all_folds_train()
        
        # fold 2 だけ実行
        result = runner.run_single_fold_train(2)
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Args:
            config: 実験設定（ExperimentConfig）
        """
        self.config = config
        self.device, self.num_gpus = get_device_and_num_gpus()
        set_seed(42)
        
        # CVSplitter を構築
        self.splitter = CVSplitter(
            splits_dict=config.cv_splits.root,
            train_ratio=config.cv_ratio.train,
            val_ratio=config.cv_ratio.val,
            test_ratio=config.cv_ratio.test,
        )
    
    # ──────────────────────────────────────────
    # 公開メソッド（train モード）
    # ──────────────────────────────────────────
    
    def run_all_folds_train(self) -> AggregatedResult:
        """
        全 fold の学習を実行する。
        
        Returns:
            AggregatedResult: 全 fold の結果
        """
        if self.config.mode != 'train':
            raise ValueError("run_all_folds_train() は mode='train' の config でのみ使用できます")
        
        results = {}
        for fold in self.splitter:
            fold_dir = self._get_fold_dir(fold.fold_idx)
            logger = setup_logging(fold_dir, f'train_fold_{fold.fold_idx}')
            
            try:
                logger.info(f"Started training for fold {fold.fold_idx}")
                result = self._run_single_fold_train_internal(fold, fold_dir, logger)
                results[fold.fold_idx] = result
                logger.info(f"Completed training for fold {fold.fold_idx}")
            except Exception as e:
                logger.error(f"Fold {fold.fold_idx} failed: {e}", exc_info=True)
                continue
        
        return AggregatedResult(fold_results=results)
    
    def run_single_fold_train(self, fold_idx: int) -> FoldResult:
        """
        指定した fold だけ学習を実行する。
        
        Args:
            fold_idx: 実行する fold のインデックス（0-based）
        
        Returns:
            FoldResult: 指定 fold の結果
        """
        if self.config.mode != 'train':
            raise ValueError("run_single_fold_train() は mode='train' の config でのみ使用できます")
        
        fold = self.splitter.get_fold(fold_idx)
        fold_dir = self._get_fold_dir(fold_idx)
        logger = setup_logging(fold_dir, f'train_fold_{fold_idx}')
        
        logger.info(f"Started training for fold {fold_idx}")
        result = self._run_single_fold_train_internal(fold, fold_dir, logger)
        logger.info(f"Completed training for fold {fold_idx}")
        
        return result
    
    # ──────────────────────────────────────────
    # 公開メソッド（test モード）
    # ──────────────────────────────────────────
    
    def run_all_folds_test(self) -> AggregatedResult:
        """
        全 fold のテストを実行する。
        
        Returns:
            AggregatedResult: 全 fold の結果
        """
        if self.config.mode != 'test':
            raise ValueError("run_all_folds_test() は mode='test' の config でのみ使用できます")
        
        # window_sizes の設定（TODO: config に移動すべき）
        window_sizes = [1, 11]
        
        all_folds_results = {}
        all_folds_window_results = WindowSizeKey.initialize_results(window_sizes)
        
        for fold_idx, fold in enumerate(self.splitter):
            fold_dir = self._get_fold_dir(fold_idx)
            logger = setup_logging(fold_dir, f'test_fold_{fold_idx}')
            
            try:
                logger.info(f"Started test for fold {fold_idx}")
                hard_multi_label_results, all_window_results = self._run_single_fold_test_internal(
                    fold_idx, fold, fold_dir, logger, window_sizes
                )
                
                # 各 fold の結果を全体の辞書に追加
                for folder_name, result in hard_multi_label_results.items():
                    if folder_name not in all_folds_results:
                        all_folds_results[folder_name] = result
                
                for window_key, sliding_window_results in all_window_results.items():
                    for folder_name, result in sliding_window_results.items():
                        all_folds_window_results[window_key][folder_name] = result
                
                logger.info(f"Completed test for fold {fold_idx}")
            except Exception as e:
                logger.error(f"Fold {fold_idx} failed: {e}", exc_info=True)
                continue
        
        # 全 fold の結果を集約
        calculator = ClassificationMetricsCalculator(
            num_classes=self.config.model.num_classes,
            mode="multitask"
        )
        
        for window_key, all_window_results in all_folds_window_results.items():
            save_dir = self.config.paths.save_dir_path / window_key
            save_dir.mkdir(parents=True, exist_ok=True)
            calculator.calculate_all_folds_metrics(all_window_results, save_dir)
        
        return AggregatedResult(fold_results={})  # TODO: 適切な結果を返す
    
    def run_single_fold_test(self, fold_idx: int) -> FoldResult:
        """
        指定した fold だけテストを実行する。
        
        Args:
            fold_idx: 実行する fold のインデックス（0-based）
        
        Returns:
            FoldResult: 指定 fold の結果
        """
        if self.config.mode != 'test':
            raise ValueError("run_single_fold_test() は mode='test' の config でのみ使用できます")
        
        # window_sizes の設定（TODO: config に移動すべき）
        window_sizes = [1, 11]
        
        fold = self.splitter.get_fold(fold_idx)
        fold_dir = self._get_fold_dir(fold_idx)
        logger = setup_logging(fold_dir, f'test_fold_{fold_idx}')
        
        logger.info(f"Started test for fold {fold_idx}")
        hard_multi_label_results, all_window_results = self._run_single_fold_test_internal(
            fold_idx, fold, fold_dir, logger, window_sizes
        )
        logger.info(f"Completed test for fold {fold_idx}")
        
        return FoldResult(fold_idx=fold_idx, metrics={})  # TODO: 適切な結果を返す
    
    # ──────────────────────────────────────────
    # 内部メソッド（train）
    # ──────────────────────────────────────────
    
    def _run_single_fold_train_internal(self, fold: FoldSplit, fold_dir: Path, logger) -> FoldResult:
        """
        1つの fold の学習を実行（内部実装）
        
        Args:
            fold: 学習する fold のデータ（FoldSplit）
            fold_dir: fold 用のディレクトリ（ログやモデル保存に使用）
            logger: ロガーオブジェクト
        
        Returns:
            FoldResult: fold の結果
        """
        
        # DataLoader 作成
        dataloader_factory = DataLoaderFactory(
            dataset_root=self.config.dataset.root,
            batch_size=self.config.dataset.batch_size,
            num_classes=self.config.model.num_classes,
            num_gpus=self.num_gpus
        )
        
        model_type = self.config.model.type  # モデルタイプを取得
        
        if model_type.value == 'single_label':
            # シングルラベル用（TODO: merge_label の設定を config に移動すべき）
            merge_label_indices = [4, 5, 6, 11, 12]
            merge_to_label = 4
            train_dataloader, val_dataloader = dataloader_factory.create_single_label_dataloaders(
                fold.train, fold.val, merge_label_indices, merge_to_label=merge_to_label
            )
        else:
            # マルチラベル用
            train_dataloader, val_dataloader = dataloader_factory.create_multi_label_dataloaders(
                fold.train, fold.val
            )
        
        # モデル構築
        model = self._build_model()
        model = model.to(self.device)
        if self.num_gpus > 1:
            model = torch.nn.DataParallel(model)
        
        # Trainer / Validator
        criterion = model_type.criterion
        optimizer = optim.Adam(model.parameters(), lr=self.config.training.learning_rate)
        trainer = Trainer(model, optimizer, criterion, self.device)
        validator = Validator(model, criterion, self.device)
        
        # 学習ループ
        loss_history = {'train': [], 'val': []}
        monitor = TrainingMonitor(fold_dir)
        
        for epoch in range(self.config.training.max_epochs):
            train_loss = trainer.train_epoch(train_dataloader)
            val_loss = validator.validate(val_dataloader)
            
            loss_history['train'].append(train_loss)
            loss_history['val'].append(val_loss)
            
            log_message = f"epoch {epoch+1}: training_loss: {train_loss:.4f} validation_loss: {val_loss:.4f}"
            logger.info(log_message)
            
            # ベストモデル保存（DataParallel の有無に関わらず module. なしで保存）
            if val_loss <= min(loss_history['val']):
                state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save(state_dict, fold_dir / "best_model.pth")
                logger.info("Best model saved.")
        
        # 学習曲線の保存
        monitor.plot_learning_curve(loss_history)
        monitor.save_loss_to_csv(loss_history)
        
        return FoldResult(fold_idx=fold.fold_idx, loss_history=loss_history)
    
    # ──────────────────────────────────────────
    # 内部メソッド（test）
    # ──────────────────────────────────────────
    
    def _run_single_fold_test_internal(
        self, 
        fold_idx: int, 
        fold: FoldSplit, 
        fold_dir: Path, 
        logger: logging.Logger, 
        window_sizes: list[int]
    ) -> tuple[dict, dict]:
        """1つの fold のテストを実行（内部実装）"""
        
        # DataLoader 作成
        dataloader_factory = DataLoaderFactory(
            dataset_root=self.config.dataset.root,
            batch_size=self.config.dataset.batch_size,
            num_classes=self.config.model.num_classes,
        )
        test_dataloaders = dataloader_factory.create_multi_label_test_dataloaders(fold.test)
        
        # モデル読み込み（DataParallel で保存された古いファイルにも対応）
        model_path = self.config.paths.model_paths_as_path[fold_idx]
        model = self._build_model()
        state_dict = torch.load(model_path, map_location=self.device)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        if self.num_gpus > 1:
            model = torch.nn.DataParallel(model)
        
        # 推論
        inference = Inference(model, self.device)
        results = inference.run(fold_dir, test_dataloaders)
        
        # 可視化・メトリクス計算（現状の test.py のロジックをそのまま使用）
        visualizer = ResultsVisualizer(fold_dir)
        converter = MultiToSingleLabelConverter(results)
        
        # マルチラベルを閾値でハードラベルに変換
        hard_multi_label_results = converter.convert_soft_to_hard_multi_labels(threshold=0.5)
        converter.save_hard_multi_label_results(hard_multi_label_results, fold_dir, methods='threshold_50%')
        
        # 可視化
        visualizer.save_main_classes_visualization(hard_multi_label_results)
        visualizer.save_multi_label_visualization(hard_multi_label_results, methods='threshold_50%')
        
        # メトリクス計算
        calculator = ClassificationMetricsCalculator(
            num_classes=self.config.model.num_classes,
            mode="multitask"
        )
        video_metrics = calculator.calculate_multi_label_metrics_per_video(hard_multi_label_results)
        overall_metrics = calculator.calculate_multi_label_overall_metrics(hard_multi_label_results)
        
        save_video_metrics_to_csv(video_metrics, fold_dir, methods='threshold_50%')
        save_overall_metrics_to_csv(overall_metrics, fold_dir / 'fold_results', methods='threshold_50%')
        
        # スライディングウィンドウ解析
        analyzer = Analyzer(fold_dir, self.config.model.num_classes)
        all_window_results = analyzer.analyze_sliding_windows(
            hard_multi_label_results, visualizer, calculator, window_sizes=window_sizes
        )
        
        # ウィンドウ解析結果を保存
        for window_key, sliding_window_results in all_window_results.items():
            visualizer.save_single_label_visualization(sliding_window_results, methods=window_key)
            
            sliding_window_video_metrics = calculator.calculate_single_label_metrics_per_video(
                sliding_window_results
            )
            sliding_window_overall_metrics = calculator.calculate_single_label_overall_metrics(
                sliding_window_results
            )
            
            save_video_metrics_to_csv(sliding_window_video_metrics, fold_dir, methods=window_key)
            save_overall_metrics_to_csv(
                sliding_window_overall_metrics,
                fold_dir / 'fold_results',
                methods=window_key
            )
        
        return hard_multi_label_results, all_window_results
    
    # ──────────────────────────────────────────
    # ヘルパーメソッド
    # ──────────────────────────────────────────
    
    def _get_fold_dir(self, fold_idx: int) -> Path:
        """fold 用のディレクトリを取得"""
        fold_dir = self.config.paths.save_dir_path / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        return fold_dir
    
    def _build_model(self):
        """config に基づいてモデルを構築"""
        if self.config.model.type.value == 'single_label':
            return SingleLabelClassificationModel(
                architecture=self.config.model.architecture,
                num_classes=self.config.model.num_classes,
                pretrained=self.config.training.pretrained if self.config.training else False,
                freeze_backbone=self.config.training.freeze_backbone if self.config.training else False,
            )
        else:
            return MultiTaskClassificationModel(
                architecture=self.config.model.architecture,
                num_classes=self.config.model.num_classes,
                pretrained=self.config.training.pretrained if self.config.training else False,
                freeze_backbone=self.config.training.freeze_backbone if self.config.training else False,
            )