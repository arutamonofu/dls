# extract_priors.py

import argparse
import json
import os
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from tqdm import tqdm

def compute_molecule_density(pos, radius=2.3):
    dists = torch.cdist(pos, pos)
    mask = (dists < radius) & (dists > 0)
    return mask.sum(dim=1).float().mean().item()

def extract_density_statistics(dataset, subset_size=1000, radius=2.3):
    loader = DataLoader(dataset[:subset_size], batch_size=1, shuffle=False)
    densities = []
    for data in tqdm(loader, desc="Вычисление плотности", leave=False):
        densities.append(compute_molecule_density(data.pos, radius))
    density_std = torch.tensor(densities).std().item()
    return {"density_std": density_std, "densities": densities}

def recommend_use_tda(density_std, threshold=0.70):
    return density_std > threshold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--subset_size', type=int, default=1000)
    parser.add_argument('--radius', type=float, default=2.3)
    parser.add_argument('--threshold', type=float, default=0.70)
    parser.add_argument('--output', type=str, default='prior_recommendation.json')
    args = parser.parse_args()

    dataset = QM9(root=args.data_path)
    stats = extract_density_statistics(dataset, args.subset_size, args.radius)
    use_tda = recommend_use_tda(stats["density_std"], args.threshold)

    result = {
        "use_tda": use_tda,
        "density_std": round(stats["density_std"], 4),
        "threshold": args.threshold,
        "radius_angstrom": args.radius,
        "subset_size": args.subset_size
    }

    output_path = os.path.join(args.data_path, 'prior_recommendation.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Стандартное отклонение плотности: {result['density_std']:.4f}")
    print(f"Рекомендация use_tda: {use_tda}")
    print(f"Результат сохранён в {output_path}")

if __name__ == '__main__':
    main()