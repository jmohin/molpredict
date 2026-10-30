"""MolGpKa's GCN, ported to plain PyTorch.

MolGpKa defines its own `GCNConv` holding `weight` and `bias` as plain Parameters rather than
using torch-geometric's, so every key in its checkpoints is fixed by its own code and does not
move with the library version. That is what makes this port possible at all, and it is the same
property that let MolCLR in; pkasolver was rejected from this project precisely because its
checkpoints were bound to torch-geometric 2.0.1's internal parameter names.

The arithmetic here is theirs, rewritten against `index_add_` so the serving image needs neither
torch-geometric nor torch-scatter. `tests/test_molgpka.py` pins outputs verified against the
reference implementation.
"""

import torch
from torch import nn

N_FEATURES = 29
# Channel widths, in order, from MolGpKa's GCNNet.
WIDTHS = [1024, 512, 256, 512, 1024]


class GCNConv(nn.Module):
    """Symmetrically normalised graph convolution with added self-loops.

    `weight` is stored (in, out) and applied as a right-multiplication, which is how MolGpKa
    wrote it; a `nn.Linear` would transpose it and the checkpoint would not load.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = x @ self.weight
        n = x.size(0)

        # Self-loops for every node, as add_remaining_self_loops gives with fill_value 1. The
        # molecule's edges are already stored in both directions, so the result is symmetric.
        loops = torch.arange(n, device=x.device)
        row = torch.cat([edge_index[0], loops])
        col = torch.cat([edge_index[1], loops])

        weights = torch.ones(row.size(0), dtype=x.dtype, device=x.device)
        degree = torch.zeros(n, dtype=x.dtype, device=x.device).index_add_(0, row, weights)
        inv_sqrt = degree.pow(-0.5)
        # An isolated node has degree 0 before self-loops; with them it cannot, but the
        # reference guards this and a guard that never fires costs nothing.
        inv_sqrt[torch.isinf(inv_sqrt)] = 0.0
        norm = inv_sqrt[row] * weights * inv_sqrt[col]

        out = torch.zeros_like(x).index_add_(0, col, norm.unsqueeze(-1) * x[row])
        return out + self.bias


class GCNNet(nn.Module):
    """Five convolutions, attention pooling, then three fully connected layers to one pKa."""

    def __init__(self) -> None:
        super().__init__()
        widths = [N_FEATURES, *WIDTHS]
        for i, (a, b) in enumerate(zip(widths, widths[1:]), start=1):
            setattr(self, f"conv{i}", GCNConv(a, b))
            setattr(self, f"bn{i}", nn.BatchNorm1d(b))
        # torch-geometric's GlobalAttention wraps the gate network under `gate_nn`, so the
        # checkpoint key is `att.gate_nn.weight`; this module reproduces that name.
        self.att = nn.Module()
        self.att.gate_nn = nn.Linear(WIDTHS[-1], 1)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i in range(1, len(WIDTHS) + 1):
            x = getattr(self, f"conv{i}")(x, edge_index)
            x = getattr(self, f"bn{i}")(torch.relu(x))

        # Attention pooling over a single graph: softmax the gate across all nodes and take the
        # weighted sum. The reference's scatter_softmax reduces to this when there is one graph.
        gate = torch.softmax(self.att.gate_nn(x), dim=0)
        pooled = (gate * x).sum(dim=0, keepdim=True)

        pooled = torch.relu(self.fc2(pooled))
        pooled = torch.relu(self.fc3(pooled))
        return self.fc4(pooled)
