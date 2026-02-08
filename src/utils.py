# src/utils.py

import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def log_mlflow_params(params):
    import mlflow
    if hasattr(params, '__dict__'):
        params = vars(params)
    
    for k, v in params.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                mlflow.log_param(f"{k}.{k2}", v2)
        else:
            mlflow.log_param(k, v)