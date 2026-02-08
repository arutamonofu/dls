# src/tda.py

import torch
import numpy as np
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy

class TDAEngine:
    def __init__(self, homology_dims=(0, 1), n_jobs=-1):
        self.vr = VietorisRipsPersistence(
            homology_dimensions=homology_dims,
            n_jobs=n_jobs
        )
        self.pe = PersistenceEntropy(
            n_jobs=n_jobs,
            normalize=True 
        )

    def compute(self, pos: torch.Tensor) -> torch.Tensor:
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"Ожидался вход (N, 3), получен {pos.shape}.")

        X = pos.detach().cpu().numpy()[None, :, :]

        diagrams = self.vr.fit_transform(X)
        features = self.pe.fit_transform(diagrams)

        return torch.tensor(features[0], dtype=torch.float32)

def get_tda_features(pos: torch.Tensor) -> torch.Tensor:
    engine = TDAEngine()
    return engine.compute(pos)