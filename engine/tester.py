from model.setup_models import setup_model
from data.dataloader import create_multilabel_test_dataloaders
from engine.inference import Inference
from evaluate.analyzer import Analyzer

class Tester:
    def __init__(self, model, device):
        self.config = config
        self.device = device
        self.num_gpus = num_gpus
        
        # self.metrics_calculator = ClassificationMetricsCalculator()
        self.inference = Inference(self.model, device)
        self.analyzer = Analyzer(self.config.paths.save_dir, self.config.test.num_classes)

    def test(self):
        # 推論
        results = self.inference.run(self.config.paths.save_dir, self.test_dataloaders)
        # 出力を解析
        self.analyzer.analyze(results)
        
        
# class Tester:
#     def __init__(self, config, device, num_gpus, test_dirs):
#         self.config = config
#         self.device = device
#         self.num_gpus = num_gpus
#         self.test_dataloaders = create_multilabel_test_dataloaders(config, test_dirs, num_gpus)
#         self.model = setup_model(config, device, num_gpus, mode='test')
#         # self.metrics_calculator = ClassificationMetricsCalculator()
#         self.inference = Inference(self.model, device)
#         self.analyzer = Analyzer(self.config.paths.save_dir, self.config.test.num_classes)

#     def test(self):
#         # 推論
#         results = self.inference.run(self.config.paths.save_dir, self.test_dataloaders)
#         # 出力を解析
#         self.analyzer.analyze(results)