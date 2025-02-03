import torch
import numpy as np
import random
from typing import Tuple

def get_device_and_num_gpus() -> Tuple[torch.device, int]:
    """Determine what device to use for PyTorch computations and return it along with the number of GPUs available."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        num_gpus = 0
    
    return device, num_gpus


def set_seed(seed: int) -> None:
    """Set the seed for all the random number generators used in the library."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
