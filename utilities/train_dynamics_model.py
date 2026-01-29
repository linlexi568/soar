"""Train a small MLP to learn quadrotor dynamics from rollout data.

This script trains f_theta(x,u) -> x_next to be used in data-driven NMPC.
The trained weights are exported to a .npz file that can be loaded by the
CasADi-based NMPC controller.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class DynamicsNet(nn.Module):
    """Small MLP for state-action to next-state prediction."""

    def __init__(self, state_dim: int = 12, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        input_dim = state_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        xu = torch.cat([x, u], dim=-1)
        return self.net(xu)


def collect_transitions_from_mock_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load state-action-next_state tuples from mock dataset rollouts (CSV format)."""
    import pandas as pd
    states, actions, next_states = [], [], []
    
    for split in ["train", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for rollout_dir in sorted(split_dir.iterdir()):
            if not rollout_dir.is_dir():
                continue
            states_file = rollout_dir / "states.csv"
            if not states_file.exists():
                continue
            
            # Load states CSV: t, px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz
            df = pd.read_csv(states_file)
            state_cols = ['px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'wx', 'wy']
            state_data = df[state_cols].values.astype(np.float32)  # (T, 12)
            
            # Create transitions (without action data, use zero actions as placeholder)
            for i in range(len(state_data) - 1):
                s = state_data[i]
                s_next = state_data[i + 1]
                a = np.zeros(4, dtype=np.float32)  # Placeholder action
                
                states.append(s)
                actions.append(a)
                next_states.append(s_next)
    
    if len(states) == 0:
        return np.zeros((0, 12)), np.zeros((0, 4)), np.zeros((0, 12))
    
    return np.array(states), np.array(actions), np.array(next_states)


def train_dynamics_model(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cuda:0",
) -> DynamicsNet:
    """Train the dynamics model using supervised learning."""
    
    X = torch.from_numpy(states).float()
    U = torch.from_numpy(actions).float()
    Y = torch.from_numpy(next_states).float()
    
    dataset = TensorDataset(X, U, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = DynamicsNet(state_dim=12, action_dim=4, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch, u_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            u_batch = u_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred = model(x_batch, u_batch)
            loss = loss_fn(pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x_batch.size(0)
        
        avg_loss = total_loss / len(dataset)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
    
    return model


def export_weights(model: DynamicsNet, output_path: Path):
    """Export model weights to .npz for CasADi."""
    state_dict = model.state_dict()
    weights = {}
    
    for i, (w_key, b_key) in enumerate([
        ("net.0.weight", "net.0.bias"),
        ("net.2.weight", "net.2.bias"),
        ("net.4.weight", "net.4.bias"),
    ]):
        weights[f"W{i}"] = state_dict[w_key].cpu().numpy()
        weights[f"b{i}"] = state_dict[b_key].cpu().numpy()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **weights)
    print(f"Exported weights to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="mock_agile_dataset")
    parser.add_argument("--output", type=str, default="results/nmpc/dynamics_weights.npz")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    print("Loading data from", args.data_dir)
    states, actions, next_states = collect_transitions_from_mock_dataset(Path(args.data_dir))
    print(f"Collected {len(states)} transitions")
    
    if len(states) == 0:
        print("No data found, exiting.")
        return
    
    print("Training dynamics model...")
    model = train_dynamics_model(
        states, actions, next_states,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    
    export_weights(model, Path(args.output))
    print("Done!")


if __name__ == "__main__":
    main()
