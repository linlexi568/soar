#!/usr/bin/env python3
"""Train Meta-RL RNN controller from collected sweep data."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader

# 添加 meta_rl 到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "meta_rl"))

from config import MetaRLConfig
from rnn_meta_policy import MetaRNNPolicy


class TraceDataset(Dataset):
    """Dataset for time-series traces of MCTS hyperparameter experiments."""
    
    def __init__(
        self,
        rows: List[Dict[str, float]],
        config: MetaRLConfig,
        window: int = 8,
    ):
        self.config = config
        self.window = window
        self.feature_names = config.feature_names
        self.output_names = config.output_names
        
        # 按 run_id 分组轨迹
        traces = defaultdict(list)
        for row in rows:
            traces[row["run_id"]].append(row)
        
        # 对每条轨迹按 iter_idx 排序
        for run_id in traces:
            traces[run_id].sort(key=lambda r: int(r["iter_idx"]))
        
        # 构建滑动窗口样本
        self.samples = []
        for run_id, trace in traces.items():
            if len(trace) < window + 1:
                continue
            
            for i in range(len(trace) - window):
                # 输入：前 window 个时间步的特征
                input_seq = []
                for t in range(i, i + window):
                    features = []
                    for fn in self.feature_names:
                        if fn == "iter_norm":
                            # 归一化：使用数据中实际最大 iter 作为分母（自适应）
                            max_iter = max(int(r["iter_idx"]) for r in trace)
                            features.append(float(trace[t]["iter_idx"]) / max(max_iter, 1.0))
                        else:
                            features.append(float(trace[t][fn]))
                    input_seq.append(features)
                
                # 目标：下一个时间步的超参数
                target_row = trace[i + window]
                targets = []
                for on in self.output_names:
                    # 映射 config 中的名称到 CSV 列名
                    if on == "dirichlet_eps":
                        targets.append(float(target_row["root_dirichlet_eps"]))
                    elif on == "dirichlet_alpha":
                        targets.append(float(target_row["root_dirichlet_alpha"]))
                    elif on == "zero_penalty" or on == "zero_action_penalty":
                        targets.append(float(target_row["zero_action_penalty"]))
                    elif on == "replicas_per_program":
                        targets.append(float(target_row["eval_replicas_per_program"]))
                    elif on == "policy_temperature":
                        # CSV 中没有此字段，使用固定值
                        targets.append(1.0)
                    else:
                        # 尝试直接从 CSV 读取
                        targets.append(float(target_row.get(on, 0.0)))
                
                self.samples.append((input_seq, targets))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_seq, targets = self.samples[idx]
        return (
            torch.tensor(input_seq, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
        )


def load_csv_data(csv_path: Path) -> List[Dict[str, float]]:
    """Load CSV file and return list of row dictionaries."""
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 转换数值字段
            row_dict = {}
            for k, v in row.items():
                try:
                    row_dict[k] = float(v) if k != "run_id" else v
                except ValueError:
                    row_dict[k] = v
            rows.append(row_dict)
    return rows


def train_epoch(
    model: MetaRNNPolicy,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs, _ = model(inputs)
        
        # 计算 MSE loss
        loss = torch.nn.functional.mse_loss(outputs, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Meta-RL RNN controller")
    parser.add_argument("--summary-csv", required=True, help="Path to collected sweep data CSV")
    parser.add_argument("--output-checkpoint", required=True, help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--window", type=int, default=8, help="Sliding window length")
    parser.add_argument("--hidden-dim", type=int, default=128, help="RNN hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of RNN layers")
    args = parser.parse_args()
    
    csv_path = Path(args.summary_csv)
    output_path = Path(args.output_checkpoint)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # 加载数据
    print(f"[train] 加载数据：{csv_path}")
    rows = load_csv_data(csv_path)
    print(f"[train] 加载了 {len(rows)} 行数据")
    
    # 统计配置数量
    unique_runs = set(row["run_id"] for row in rows)
    print(f"[train] 发现 {len(unique_runs)} 个独特配置")
    
    # 创建配置和数据集
    config = MetaRLConfig()
    dataset = TraceDataset(rows, config, window=args.window)
    print(f"[train] 生成了 {len(dataset)} 个训练样本（滑动窗口）")
    
    if len(dataset) == 0:
        raise ValueError("没有足够的数据生成训练样本！请检查 CSV 文件。")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] 使用设备：{device}")
    
    # 更新 config 中的参数
    config.hidden_dim = args.hidden_dim
    config.num_layers = args.num_layers
    
    model = MetaRNNPolicy(config)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练循环
    print(f"[train] 开始训练：{args.epochs} 轮")
    print(f"[train] 批量大小：{args.batch_size}，学习率：{args.lr}")
    print("-" * 60)
    
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, dataloader, optimizer, device)
        print(f"[Epoch {epoch:3d}] loss={loss:.4f}")
    
    # 保存模型
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "window": args.window,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
    }, output_path)
    
    print("-" * 60)
    print(f"[train] ✓ 训练完成！模型已保存到：{output_path}")


if __name__ == "__main__":
    main()
