"""Fine-tune MolCLR's pretrained GIN encoder on one OPERA endpoint.

    uv run python train.py boiling_point

Writes runs/<endpoint>/model.pt (weights, target scaling and the metrics it earned) and prints
held-out performance on a scaffold split.

Deliberate departures from MolCLR's own finetune.py, each of which would otherwise cost us a
correct result:
  * It catches only FileNotFoundError when loading the checkpoint and otherwise prints
    "Training from scratch", so a bad path yields a randomly initialised model that still
    reports a score. Here a short checkpoint is a hard error.
  * It only z-scores targets for two named QM datasets. Boiling point in degC against an MSE
    loss needs it just as much, so every regression target is standardised.
  * Its split is unseeded, so runs are not comparable.
"""

import csv
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))
from molpredict_backend.vendor.molclr.featurize import Unfeaturizable, featurize_smiles  # noqa: E402
from molpredict_backend.vendor.molclr.model import GINet  # noqa: E402

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "backend/src/molpredict_backend/vendor/molclr/pretrained_gin.pth"
)
# The encoder has 59 tensors; the checkpoint's other 4 belong to the pretraining head.
EXPECTED_PRETRAINED_TENSORS = 59

SEED = 20261029
VALID_FRACTION = 0.1
TEST_FRACTION = 0.1


@dataclass
class Graph:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: float
    smiles: str


def load(endpoint: str) -> list[Graph]:
    path = HERE / "data" / f"{endpoint}.csv"
    graphs, skipped = [], 0
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                x, edge_index, edge_attr = featurize_smiles(row["smiles"])
            except Unfeaturizable:
                skipped += 1
                continue
            graphs.append(
                Graph(
                    torch.from_numpy(x),
                    torch.from_numpy(edge_index),
                    torch.from_numpy(edge_attr),
                    float(row["target"]),
                    row["smiles"],
                )
            )
    if skipped:
        print(f"  skipped {skipped} molecules MolCLR cannot represent")
    return graphs


def scaffold_split(graphs: list[Graph]) -> tuple[list[int], list[int], list[int]]:
    """Split by Bemis-Murcko scaffold, as MolCLR does.

    Molecules sharing a scaffold go to the same side, so the test set measures generalisation
    to new chemistry rather than to near-copies of the training set. A random split flatters
    the model badly here.
    """
    scaffolds: dict[str, list[int]] = {}
    for i, graph in enumerate(graphs):
        mol = Chem.MolFromSmiles(graph.smiles)
        try:
            key = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            key = ""
        if not key:
            # An acyclic molecule has no ring scaffold, and Murcko returns "" for every one of
            # them. Treating that as a shared scaffold puts methane, ethanol and hexadecane in
            # one group — half this dataset for boiling point — which then lands wholly on one
            # side of the split. They are not near-duplicates of each other, so each stands
            # alone.
            key = f"acyclic:{graph.smiles}"
        scaffolds.setdefault(key, []).append(i)

    # Shuffling rather than sorting by size. Largest-first assignment steers big scaffold
    # families into training and leaves the test set made of whatever is rarest, so the test
    # population differs from the training one by construction and the score measures that
    # difference as much as the model. Shuffling keeps the guarantee that matters — no scaffold
    # spans two splits — while leaving the splits chemically comparable.
    # Biggest groups first, so the ones that cannot be subdivided are placed while every split
    # still has room. A shuffled order lets a late arrival overshoot: one benzene scaffold
    # covers a third of the pKa set, and landing it last leaves valid and test far short.
    groups = sorted(scaffolds.values(), key=lambda g: (-len(g), g[0]))

    n = len(graphs)
    quotas = {
        "train": (1.0 - VALID_FRACTION - TEST_FRACTION) * n,
        "valid": VALID_FRACTION * n,
        "test": TEST_FRACTION * n,
    }
    splits: dict[str, list[int]] = {"train": [], "valid": [], "test": []}

    # Each group goes wherever there is most room left. MolCLR's own version walks the groups
    # filling train, then valid, then test against running cutoffs, but train and valid fill at
    # the same time there, so the test branch can simply never be reached — it handed back an
    # empty test set on this data.
    for group in groups:
        name = max(splits, key=lambda k: quotas[k] - len(splits[k]))
        splits[name] += group
    return splits["train"], splits["valid"], splits["test"]


def collate(batch: list[Graph]) -> tuple[torch.Tensor, ...]:
    xs, edges, attrs, batch_index = [], [], [], []
    offset = 0
    for i, graph in enumerate(batch):
        xs.append(graph.x)
        edges.append(graph.edge_index + offset)
        attrs.append(graph.edge_attr)
        batch_index.append(torch.full((graph.x.size(0),), i, dtype=torch.long))
        offset += graph.x.size(0)
    return (
        torch.cat(xs),
        torch.cat(edges, dim=1),
        torch.cat(attrs),
        torch.cat(batch_index),
        torch.tensor([g.y for g in batch], dtype=torch.float32),
    )


def batches(graphs: list[Graph], indices: list[int], size: int, shuffle: bool):
    order = list(indices)
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(order), size):
        chunk = [graphs[i] for i in order[start : start + size]]
        # BatchNorm needs more than one graph while training.
        if shuffle and len(chunk) == 1:
            continue
        yield collate(chunk)


def evaluate(model, graphs, indices, task, mean, std, size=128):
    model.eval()
    preds, actual = [], []
    with torch.no_grad():
        for x, edge_index, edge_attr, batch, y in batches(graphs, indices, size, False):
            _, out = model(x, edge_index, edge_attr, batch)
            if task == "classification":
                preds += torch.softmax(out, dim=1)[:, 1].tolist()
            else:
                preds += (out.squeeze(-1) * std + mean).tolist()
            actual += y.tolist()
    return np.array(preds), np.array(actual)


def metrics(preds, actual, task) -> dict[str, float]:
    if task == "classification":
        order = np.argsort(preds)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(preds) + 1)
        pos, neg = actual == 1, actual == 0
        n_pos, n_neg = pos.sum(), neg.sum()
        auroc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        accuracy = ((preds > 0.5).astype(float) == actual).mean()
        return {"AUROC": float(auroc), "accuracy": float(accuracy)}
    errors = preds - actual
    ss_res = float((errors**2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    return {
        "RMSE": float(np.sqrt((errors**2).mean())),
        "MAE": float(np.abs(errors).mean()),
        "R^2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
    }


def train(endpoint: str, task: str, epochs: int = 100, batch_size: int = 32) -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"=== {endpoint} ({task}) ===")
    graphs = load(endpoint)
    train_idx, valid_idx, test_idx = scaffold_split(graphs)
    print(f"  {len(graphs)} molecules -> train {len(train_idx)} / valid {len(valid_idx)} / test {len(test_idx)}")

    if task == "regression":
        values = [graphs[i].y for i in train_idx]
        mean, std = statistics.fmean(values), statistics.pstdev(values) or 1.0
    else:
        mean, std = 0.0, 1.0
    if task == "regression":
        for g in graphs:
            g.y = (g.y - mean) / std

    model = GINet(task=task, drop_ratio=0.3)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    adopted = model.load_pretrained(state)
    if adopted != EXPECTED_PRETRAINED_TENSORS:
        raise SystemExit(
            f"Only {adopted} pretrained tensors loaded, expected "
            f"{EXPECTED_PRETRAINED_TENSORS}. Refusing to train on a half-initialised encoder."
        )
    print(f"  loaded {adopted} pretrained tensors")

    head = [p for n, p in model.named_parameters() if n.startswith("pred_head")]
    base = [p for n, p in model.named_parameters() if not n.startswith("pred_head")]
    optimizer = torch.optim.Adam(
        [{"params": base, "lr": 1e-4}, {"params": head, "lr": 5e-4}], weight_decay=1e-6
    )
    # Decaying to near zero lets the last epochs settle instead of bouncing around the minimum.
    # Measured, not assumed: on the four endpoints whose data did not change it made three
    # slightly worse (both pKa models and biodegradability) and one marginally better, so it is
    # no improvement. It stays only because it is what produced the three checkpoints currently
    # shipped, and removing it would stop this file reproducing them.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

    best_score, best_state, best_epoch = math.inf, None, -1
    started = time.time()
    for epoch in range(epochs):
        model.train()
        for x, edge_index, edge_attr, batch, y in batches(graphs, train_idx, batch_size, True):
            optimizer.zero_grad()
            _, out = model(x, edge_index, edge_attr, batch)
            loss = (
                criterion(out, y.long())
                if task == "classification"
                else criterion(out.squeeze(-1), y)
            )
            loss.backward()
            optimizer.step()

        scheduler.step()

        preds, actual = evaluate(model, graphs, valid_idx, task, mean, std)
        if task == "regression":
            actual = actual * std + mean
        score = (
            -metrics(preds, actual, task)["AUROC"]
            if task == "classification"
            else metrics(preds, actual, task)["RMSE"]
        )
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  valid {'AUROC' if task == 'classification' else 'RMSE'} {abs(score):.3f}  ({time.time() - started:.0f}s)")

    model.load_state_dict(best_state)
    preds, actual = evaluate(model, graphs, test_idx, task, mean, std)
    if task == "regression":
        actual = actual * std + mean
    result = metrics(preds, actual, task)
    print(f"  best epoch {best_epoch}; held-out scaffold test: " + ", ".join(f"{k} {v:.3f}" for k, v in result.items()))

    out_dir = HERE / "runs" / endpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "task": task,
            "target_mean": mean,
            "target_std": std,
            "metrics": result,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        },
        out_dir / "model.pt",
    )
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"  saved {out_dir / 'model.pt'}")


TASKS = {
    "boiling_point": "regression",
    "melting_point": "regression",
    "vapor_pressure": "regression",
    "ready_biodegradability": "classification",
    "rat_oral_ld50": "regression",
    # From the IUPAC dataset, which is CC BY-NC — see prepare_pka.py.
    "pka_acidic": "regression",
    "pka_basic": "regression",
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(TASKS)
    for name in names:
        train(name, TASKS[name])
