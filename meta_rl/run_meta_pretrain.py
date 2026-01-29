from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader

# 修复：直接导入而非相对导入
try:
    from .config import MetaRLConfig
    from .rnn_meta_policy import MetaRNNPolicy
except ImportError:
    from config import MetaRLConfig
    from rnn_meta_policy import MetaRNNPolicy


class TraceDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, float]],
        config: MetaRLConfig,
        window: int = 8,
    ) -> None:
        self.config = config
        self.window = window
        self.samples: List[Dict[str, torch.Tensor]] = []
        runs = defaultdict(list)
        for row in rows:
            run_id = row.get("run_id") or row.get("config_id") or "default"
            runs[run_id].append(row)
        for run_rows in runs.values():
            run_rows.sort(key=lambda r: float(r.get("iter_idx", len(run_rows))))
            feat_stack: List[List[float]] = []
            for entry in run_rows:
                features = [float(entry.get(name, 0.0)) for name in config.feature_names]
                feat_stack.append(features)
                if len(feat_stack) >= window:
                    seq = torch.tensor(feat_stack[-window:], dtype=torch.float32)
                    target = []
                    for name in config.output_names:
                        key = f"target_{name}"
                        if key in entry:
                            target.append(float(entry[key]))
                        else:
                            target.append(float(entry.get(name, 0.0)))
                    target_tensor = torch.tensor(target, dtype=torch.float32)
                    self.samples.append({
                        "sequence": seq,
                        "target": target_tensor,
                    })
        if not self.samples:
            raise ValueError("Dataset is empty. Check that CSV contains enough rows.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        return sample["sequence"], sample["target"]


def load_rows(csv_path: Path) -> List[Dict[str, float]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            cleaned = {}
            for k, v in row.items():
                if v in (None, ""):
                    cleaned[k] = 0.0
                    continue
                try:
                    cleaned[k] = float(v)
                except ValueError:
                    cleaned[k] = v
            rows.append(cleaned)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MetaRLConfig()
    rows = load_rows(Path(args.summary_csv))
    dataset = TraceDataset(rows, config, window=args.window)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    policy = MetaRNNPolicy(config).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for sequences, targets in loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            outputs, _ = policy(sequences)
            loss = mse(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        print(f"[Epoch {epoch}] loss={epoch_loss:.4f}")

    checkpoint_path = Path(args.output_checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy_state": policy.state_dict(), "config": config.__dict__}, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline pretraining for Meta-RL controller")
    parser.add_argument("--summary-csv", required=True, help="Path to CSV summary logs")
    parser.add_argument("--output-checkpoint", default="meta_rl/checkpoints/meta_policy.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--window", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
