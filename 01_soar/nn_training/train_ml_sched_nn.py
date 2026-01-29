"""Train a tiny NN for NNScheduler from CSV logs and export TorchScript.

Usage example (PowerShell):
    .venv\\Scripts\\python.exe 01_soar\\nn_training\\train_ml_sched_nn.py \
        --csv 01_soar\\results\\ml_sched_samples.csv \
        --out 01_soar\\results\\nn_trained\\ml_sched.pt
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud


# Training features must match the runtime NNScheduler input (7 dims):
# progress, best_reward, best_reward_delta, seconds_since_improve,
# iters_since_improve, rule_count, epsilon
IN_FEATS = [
    'progress','best_reward','best_reward_delta',
    'seconds_since_improve','iters_since_improve','rule_count','epsilon'
]


def load_csv(path: str, key_order: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, 'r', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        X: List[List[float]] = []
        Y: List[List[float]] = []
        for row in rdr:
            try:
                x = [float(row[k]) for k in IN_FEATS]
                y = [float(row[k]) for k in key_order]
            except Exception:
                # skip malformed lines
                continue
            X.append(x); Y.append(y)
    if not X:
        raise RuntimeError(f"No valid samples in CSV: {path}")
    Xn = np.asarray(X, dtype=np.float32)
    Yn = np.asarray(Y, dtype=np.float32)
    return Xn, Yn


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, required=True, help='由训练脚本 --ml-dump-csv 导出的样本文件')
    p.add_argument('--out', type=str, default='01_soar/results/nn_trained/ml_sched.pt', help='导出的 TorchScript 路径')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--val-frac', type=float, default=0.1)
    p.add_argument('--device', type=str, default='cpu')
    args = p.parse_args()

    # Import shared KEY_ORDER to align targets via compatibility shim
    try:
        from ..ml_param_scheduler import KEY_ORDER  # type: ignore
    except Exception:
        from ..mcts_training.ml_param_scheduler import KEY_ORDER  # type: ignore

    X, Y = load_csv(args.csv, KEY_ORDER)
    N = X.shape[0]
    val_n = int(max(1, min(N-1, round(N * float(args.val_frac)))))
    idx = np.arange(N)
    rng = np.random.default_rng(123)
    rng.shuffle(idx)
    val_idx = idx[:val_n]; tr_idx = idx[val_n:]

    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xva, Yva = X[val_idx], Y[val_idx]

    tr_ds = tud.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
    va_ds = tud.TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva))
    tr_ld = tud.DataLoader(tr_ds, batch_size=args.bs, shuffle=True)
    va_ld = tud.DataLoader(va_ds, batch_size=args.bs, shuffle=False)

    model = TinyMLP(X.shape[1], Y.shape[1]).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.SmoothL1Loss()

    best_state = None
    best_val = float('inf')
    for ep in range(1, args.epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_ld:
            xb = xb.to(args.device); yb = yb.to(args.device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= max(1, len(tr_ds))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_ld:
                xb = xb.to(args.device); yb = yb.to(args.device)
                pred = model(xb)
                loss = crit(pred, yb)
                va_loss += loss.item() * xb.size(0)
        va_loss /= max(1, len(va_ds))
        print(f"[ml-sched-train] epoch {ep:03d} | train {tr_loss:.6f} | val {va_loss:.6f}")
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Wrap to match expected NNScheduler input shape
    class Wrapper(nn.Module):
        def __init__(self, core: nn.Module):
            super().__init__()
            self.core = core
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 1:
                x = x.unsqueeze(0)
                y = self.core(x)
                return y.squeeze(0)
            return self.core(x)

    ts = torch.jit.script(Wrapper(model).eval())
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ts.save(args.out)
    print(f"[ml-sched-train] saved TorchScript to {args.out}")


if __name__ == '__main__':
    main()
