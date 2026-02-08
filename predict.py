# predict.py

import argparse
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.dataset import load_qm9, get_dataloaders
from src.model import HybridEGNN
from src.utils import get_device, seed_everything

def predict(args: argparse.Namespace):
    seed_everything(args.seed)
    device = get_device()
    
    if not os.path.exists(args.model_path):
        print(f"Файл модели не найден по пути: {args.model_path}")
        return

    dataset = load_qm9(root=args.data_path, tda_features_path=args.tda_path if args.use_tda else None, target_idx=args.target_idx)
    _, _, test_loader = get_dataloaders(dataset, batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers)

    train_loader, _, _ = get_dataloaders(dataset, batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers)
    
    ys_list = [batch.y for batch in tqdm(train_loader, desc="Расчёт статистик", leave=False)]
    train_ys = torch.cat(ys_list, dim=0)
    mean, std = train_ys.mean(), train_ys.std()

    sample = next(iter(train_loader))
    tda_dim = sample.tda_x.shape[-1] if hasattr(sample, 'tda_x') else 0

    model = HybridEGNN(args.hidden_dim, args.hidden_dim, n_egnn_layers=args.n_layers, tda_feature_dim=tda_dim).to(device)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"Не удалось загрузить веса: {e}.")
        return

    model.eval()

    all_preds, all_trues = [], []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Предсказание"):
            data = data.to(device)
            tda_x = data.tda_x if args.use_tda else None

            pred_norm = model(data.z, data.pos, data.batch, data.edge_index, tda_x)
            
            pred_real = (pred_norm.float().cpu() * std) + mean
            all_preds.append(pred_real.numpy())
            all_trues.append(data.y.cpu().numpy())

    preds, trues = np.concatenate(all_preds).flatten(), np.concatenate(all_trues).flatten()
    df = pd.DataFrame({'y_true': trues, 'y_pred': preds, 'error': np.abs(trues - preds)})
    
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    df.to_csv(args.output, index=False)
    print(f"Готово. MAE: {df['error'].mean():.4f}. Результаты в {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--target_idx', type=int, required=True)
    parser.add_argument('--use_tda', action='store_true')
    parser.add_argument('--output', type=str, default='predictions.csv')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--tda_path', type=str, default='data/tda_features.pt')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    predict(args)