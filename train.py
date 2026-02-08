# train.py

import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

from src.dataset import load_qm9, get_dataloaders
from src.model import HybridEGNN
from src.utils import seed_everything, get_device, count_parameters

def train_epoch(model, loader, optimizer, loss_fn, device, std):
    model.train()
    total_loss = 0

    for data in tqdm(loader, desc="Обучение", leave=False):
        data = data.to(device)
        optimizer.zero_grad()

        tda_x = data.tda_x if hasattr(data, 'tda_x') else None
        pred = model(data.z, data.pos, data.batch, data.edge_index, tda_x)
        
        loss = loss_fn(pred, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)      
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return (total_loss / len(loader.dataset)) * std.item()

def evaluate(model, loader, loss_fn, device, std):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(loader, desc="Оценка", leave=False):
            data = data.to(device)
            tda_x = data.tda_x if hasattr(data, 'tda_x') else None

            pred = model(data.z, data.pos, data.batch, data.edge_index, tda_x)
            loss = loss_fn(pred, data.y)
            
            if not torch.isnan(loss):
                total_loss += loss.item() * data.num_graphs
                
    return (total_loss / len(loader.dataset)) * std.item()

def main(args):
    seed_everything(args.seed)
    device = get_device()
    os.makedirs("models", exist_ok=True)
    
    target_label = "U0" if args.target_idx == 7 else "Gap" if args.target_idx == 4 else f"idx{args.target_idx}"
    mlflow.set_experiment(f"QM9-Target-{target_label}")

    dataset = load_qm9(root=args.data_path, tda_features_path=args.tda_path if args.use_tda else None, target_idx=args.target_idx)
    train_loader, val_loader, test_loader = get_dataloaders(dataset, args.batch_size, seed=args.seed, num_workers=args.num_workers)

    ys_list = [batch.y for batch in tqdm(train_loader, desc="Сбор статистики", leave=False)]
    train_ys = torch.cat(ys_list, dim=0)
    mean, std = train_ys.mean(), train_ys.std()
    
    target_attr = dataset._data if hasattr(dataset, '_data') else dataset.data
    target_attr.y = (target_attr.y - mean) / std

    model = HybridEGNN(
        node_feature_dim=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        n_egnn_layers=args.n_layers,
        tda_feature_dim=next(iter(train_loader)).tda_x.shape[-1] if args.use_tda else 0
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)
    loss_fn = F.l1_loss
    
    if args.n_layers == 0:
        mode = "Baseline_FCNN"
    else:
        mode = "Hybrid_TDA" if args.use_tda else "Base_EGNN"
    
    current_run_name = f"{target_label}_{mode}_seed{args.seed}"
    model_filename = f'models/{current_run_name}.pth'

    with mlflow.start_run(run_name=current_run_name):
        mlflow.log_params(vars(args))
        mlflow.log_param("mean", mean.item())
        mlflow.log_param("std", std.item())

        best_val_mae = float('inf')
        model_saved = False

        for epoch in range(1, args.epochs + 1):
            train_mae = train_epoch(model, train_loader, optimizer, loss_fn, device, std)
            val_mae = evaluate(model, val_loader, loss_fn, device, std)
            
            if not torch.isnan(torch.tensor(val_mae)):
                scheduler.step(val_mae)

            mlflow.log_metrics({"train_mae": train_mae, "val_mae": val_mae}, step=epoch)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), model_filename)
                model_saved = True
                mlflow.log_metric("best_val_mae", best_val_mae, step=epoch)

            if epoch % 10 == 0 or epoch == 1:
                print(f"Эпоха {epoch:03d} | Train: {train_mae:.4f} | Val: {val_mae:.4f}")

        if not model_saved:
            print(f"Лучшая модель не была сохранена. Сохраняем текущую.")
            torch.save(model.state_dict(), model_filename)

        if os.path.exists(model_filename):
            model.load_state_dict(torch.load(model_filename, weights_only=True))
        
        test_mae = evaluate(model, test_loader, loss_fn, device, std)
        mlflow.log_metric("final_test_mae", test_mae)
        mlflow.log_artifact(model_filename)
        print(f"\nИтоговый Test MAE: {test_mae:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--tda_path', type=str, default='data/tda_features.pt')
    parser.add_argument('--use_tda', action='store_true')
    parser.add_argument('--target_idx', type=int, default=7)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    main(args)