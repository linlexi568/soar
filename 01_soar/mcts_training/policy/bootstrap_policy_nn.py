"""Bootstrap a constant-output PolicyNN checkpoint for quick enablement.

If torch is available, initializes a tiny network with near-uniform outputs.
Saves to 01_soar/results/policynn.pt by default.
"""
from __future__ import annotations
import os
from typing import Optional

try:
    import torch
    TORCH = True
except Exception:
    TORCH = False

from .policy_nn import PolicyNNModel, EDIT_TYPES, get_default_device

def main(out_path: Optional[str] = None):
    if out_path is None:
        out_path = os.path.join('01_soar', 'results', 'policynn.pt')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    if not TORCH:
        # Write a tiny marker file so trainer knows no torch weights
        with open(out_path + '.NO_TORCH.txt', 'w', encoding='utf-8') as f:
            f.write('PyTorch not available; using heuristic/JSON prior.')
        print(f"[PolicyNN] Torch not available, wrote marker: {out_path}.NO_TORCH.txt")
        return
    device = get_default_device('cpu')
    model = PolicyNNModel()
    model.to(device)
    # Initialize with small weights so outputs are near uniform after softmax
    for p in model.parameters():  # type: ignore[attr-defined]
        if hasattr(p, 'data'):
            p.data.zero_()
    if TORCH:
        import torch as _torch  # type: ignore
        _torch.save(model.state_dict(), out_path)  # type: ignore[attr-defined]
    print(f"[PolicyNN] Bootstrapped constant model -> {out_path} (types={len(EDIT_TYPES)})")

if __name__ == '__main__':
    main()
