# src/dataset.py

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

def load_qm9(root: str, tda_features_path: str = None, target_idx: int = None):
    dataset = QM9(root=root)
    if tda_features_path:
        tda_tensor = torch.load(tda_features_path, weights_only=True)
        if len(tda_tensor) != len(dataset):
            raise ValueError(f"Длина TDA-признаков отличается.")
        
        target = dataset._data
        target.tda_x = tda_tensor.float()
        dataset.slices['tda_x'] = torch.arange(len(dataset) + 1, dtype=torch.long)

    if target_idx is not None:
        target = dataset._data if hasattr(dataset, '_data') else dataset.data
        target.y = target.y[:, target_idx].unsqueeze(1)
    return dataset

def get_dataloaders(dataset, batch_size: int = 64, split: tuple = (0.8, 0.1, 0.1), seed: int = 42, num_workers: int = 4):
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)
    
    total = len(dataset)
    n_train = int(total * split[0])
    n_val = int(total * split[1])

    kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True}

    train_loader = DataLoader(dataset[indices[:n_train]], shuffle=True, **kwargs)
    val_loader = DataLoader(dataset[indices[n_train : n_train + n_val]], shuffle=False, **kwargs)
    test_loader = DataLoader(dataset[indices[n_train + n_val :]], shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader