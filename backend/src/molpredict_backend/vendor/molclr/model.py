"""MolCLR's GIN encoder, ported to plain PyTorch.

Wang, Y.; Wang, J.; Cao, Z.; Barati Farimani, A. Molecular contrastive learning of
representations via graph neural networks. Nature Machine Intelligence 2022, 4, 279-287.
Code MIT licensed, (c) 2021 Yuyang Wang — see LICENSE.txt.

The original builds on torch-geometric's MessagePassing. That layer holds no learnable
parameters, so the published checkpoint's keys come from the module names below and nothing
else: writing the message passing out by hand keeps the weights loadable and keeps
torch-geometric out of the serving image. The port is checked against the reference
implementation numerically rather than on the strength of that argument.

Module names here must stay exactly as they are, or the checkpoint stops matching.
"""

import torch
from torch import nn

from .featurize import SELF_LOOP_BOND_TYPE

NUM_ATOM_TYPE = 119
NUM_CHIRALITY_TAG = 3
NUM_BOND_TYPE = 5
NUM_BOND_DIRECTION = 3


def global_mean_pool(h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Average node features within each graph of a batch."""
    size = int(batch.max()) + 1 if batch.numel() else 0
    out = torch.zeros(size, h.size(1), dtype=h.dtype, device=h.device)
    out.index_add_(0, batch, h)
    counts = torch.zeros(size, dtype=h.dtype, device=h.device)
    counts.index_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
    return out / counts.clamp(min=1).unsqueeze(-1)


class GINEConv(nn.Module):
    """One GIN layer: a node takes the sum of its neighbours plus each bond's embedding."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        num_nodes = x.size(0)

        # Self-loops let a node keep its own state; they are appended, as add_self_loops does.
        loop = torch.arange(num_nodes, device=x.device)
        edge_index = torch.cat([edge_index, loop.unsqueeze(0).repeat(2, 1)], dim=1)
        loop_attr = torch.zeros(num_nodes, 2, dtype=edge_attr.dtype, device=x.device)
        loop_attr[:, 0] = SELF_LOOP_BOND_TYPE
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        # Default message passing is source-to-target: row 0 sends, row 1 receives.
        messages = x[edge_index[0]] + edge_embeddings
        aggregated = torch.zeros_like(x).index_add_(0, edge_index[1], messages)
        return self.mlp(aggregated)


class GINet(nn.Module):
    def __init__(
        self,
        task: str = "regression",
        num_layer: int = 5,
        emb_dim: int = 300,
        feat_dim: int = 512,
        drop_ratio: float = 0.0,
        pred_n_layer: int = 2,
    ) -> None:
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)

        self.gnns = nn.ModuleList(GINEConv(emb_dim) for _ in range(num_layer))
        self.batch_norms = nn.ModuleList(nn.BatchNorm1d(emb_dim) for _ in range(num_layer))
        self.feat_lin = nn.Linear(emb_dim, feat_dim)

        out_dim = 1 if task == "regression" else 2
        # Softplus head, the reference default. Its `relu` branch appends the output layer
        # twice; this follows the branch the published configs actually use.
        head: list[nn.Module] = [nn.Linear(feat_dim, feat_dim // 2), nn.Softplus()]
        for _ in range(max(1, pred_n_layer) - 1):
            head += [nn.Linear(feat_dim // 2, feat_dim // 2), nn.Softplus()]
        head.append(nn.Linear(feat_dim // 2, out_dim))
        self.pred_head = nn.Sequential(*head)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = nn.functional.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = nn.functional.dropout(
                    nn.functional.relu(h), self.drop_ratio, training=self.training
                )

        h = global_mean_pool(h, batch)
        h = self.feat_lin(h)
        return h, self.pred_head(h)

    def load_pretrained(self, state_dict: dict[str, torch.Tensor]) -> int:
        """Copy the pretrained encoder in, leaving the prediction head freshly initialised.

        Returns how many tensors were adopted. The reference silently falls back to a random
        init when the checkpoint is missing, producing a model that trains and reports results
        while having learned nothing from pretraining — so callers check this count.
        """
        own = self.state_dict()
        adopted = 0
        for name, param in state_dict.items():
            if name not in own:
                continue
            if own[name].shape != param.shape:
                raise ValueError(
                    f"Checkpoint tensor {name} is {tuple(param.shape)}, "
                    f"model wants {tuple(own[name].shape)}."
                )
            own[name].copy_(param)
            adopted += 1
        return adopted
