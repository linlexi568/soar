"""Bootstrap a constant-output NNScheduler checkpoint for quick enablement.

Saves to 01_soar/results/nn_trained/ml_sched.pt by default.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the package is on the path
_p = str(Path(__file__).resolve().parents[1])
if _p not in sys.path:
    sys.path.insert(0, _p)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]


def main():
    """Generate and save a minimal TorchScript model."""
    out_dir = os.path.join('01_soar', 'results', 'nn_trained')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ml_sched.pt')

    if not TORCH_AVAILABLE:
        # Create a marker file if torch is not available
        with open(f"{out_path}.NO_TORCH.txt", "w") as f:
            f.write('PyTorch not available; using heuristic/JSON prior.')
        print(f"[NNScheduler] Torch not available, wrote marker: {out_path}.NO_TORCH.txt")
        return

    # --- Define the minimal model ---
    IN_DIM = 7
    OUT_DIM = 11
    DEFAULTS = torch.tensor([  # type: ignore[union-attr]
        0.82, 0.82, 1.10, 1.00, 0.15, 0.10, 0.55, 0.00, 0.02, 6.00, 0.10, 0.25
    ][:OUT_DIM], dtype=torch.float32)  # type: ignore[union-attr]

    class SchedTiny(nn.Module):  # type: ignore[misc]
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(  # type: ignore[union-attr]
                nn.Linear(IN_DIM, 16), nn.ReLU(),  # type: ignore[union-attr]
                nn.Linear(16, OUT_DIM)  # type: ignore[union-attr]
            )
            with torch.no_grad():  # type: ignore[union-attr]
                for m in self.fc.modules():
                    if isinstance(m, nn.Linear):  # type: ignore[union-attr]
                        nn.init.zeros_(m.weight)  # type: ignore[union-attr]
                        nn.init.zeros_(m.bias)    # type: ignore[union-attr]

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':  # type: ignore[name-defined]
            y = self.fc(x)
            if x.ndim == 1:
                return y + DEFAULTS
            return y + DEFAULTS.unsqueeze(0)

    model = SchedTiny().eval()  # type: ignore[operator]
    scripted_model = torch.jit.script(model)  # type: ignore[union-attr]
    scripted_model.save(out_path)  # type: ignore[union-attr]
    print(f"[NNScheduler] Bootstrapped constant model -> {out_path}")


if __name__ == '__main__':
    main()
