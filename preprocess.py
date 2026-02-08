# preprocess.py

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import torch
import numpy as np
import warnings
from tqdm import tqdm
from src.dataset import load_qm9
from src.tda import TDAEngine

warnings.filterwarnings("ignore")
np.seterr(all='ignore')

ROOT_DIR = "data"
OUTPUT_PATH = os.path.join(ROOT_DIR, "tda_features.pt")

def main():
    dataset = load_qm9(root=ROOT_DIR, tda_features_path=None)
    engine = TDAEngine(homology_dims=(0, 1), n_jobs=-1)
    
    all_positions = [data.pos.numpy() for data in tqdm(dataset, desc="Сбор координат")]

    chunk_size = 10000
    all_features = []
    
    for i in range(0, len(all_positions), chunk_size):
        chunk = all_positions[i : i + chunk_size]

        with np.errstate(divide='ignore', invalid='ignore'):
            diagrams = engine.vr.fit_transform(chunk)
            features = engine.pe.fit_transform(diagrams)
        
        all_features.append(torch.tensor(features, dtype=torch.float32))
        print(f"Обработано: {min(i + chunk_size, len(all_positions))}/{len(all_positions)}")

    final_features = torch.cat(all_features, dim=0)
    torch.save(final_features, OUTPUT_PATH)
    print(f"TDA-признаки {final_features.shape} сохранены в {OUTPUT_PATH}")

if __name__ == "__main__":
    main()