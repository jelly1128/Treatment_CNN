import torch.nn as nn
from .cnn_models import MultiLabelDetectionModel

def setup_model(config, device, num_gpus):
    """モデルをGPUに移動し、並列化"""
    model = MultiLabelDetectionModel(num_classes=config.training.num_classes,
                                     pretrained=config.training.pretrained,
                                     freeze_backbone=config.training.freeze_backbone)
    
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs for training")
        model = nn.DataParallel(model)  # または DistributedDataParallel

    model = model.to(device)
    
    return model