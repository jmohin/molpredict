"""Build acidic and basic pKa training sets from the IUPAC Digitized pKa Dataset.

    uv run python prepare_pka.py

LICENSING — this source is not like the others.

  Zheng, Jonathan W. (2022) IUPAC Digitized pKa Dataset, v1.0.
  Copyright (c) 2022 International Union of Pure and Applied Chemistry.
  Reproduced by permission of IUPAC. Licensed CC BY-NC 4.0.
  https://doi.org/10.5281/zenodo.7236453

CC BY-NC forbids commercial use. Everything else in molpredict is Apache-2.0 / MIT / BSD, so
models trained from this data carry a restriction the rest of the project does not. The data is
downloaded at training time and never committed, but the resulting weights are a derivative
work: keep the attribution with them, and do not put them behind a paywall.

Why two targets rather than one "pKa": a molecule has a pKa per ionisable site, and the value
for an acid losing a proton is a different quantity from that for a base gaining one. Asking a
model for a single number per molecule is the mistake that left pKaPredict barely better than
guessing a constant. `pK1` and `pKAH1` are each well defined, so each gets its own model.
"""

import csv
import io
import statistics
import sys
import urllib.request
from pathlib import Path

from rdkit import RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))
from molpredict_backend import chem  # noqa: E402

RDLogger.DisableLog("rdApp.*")

SOURCE = (
    "https://raw.githubusercontent.com/jonwzheng/Dissociation-Constants-2/main/"
    "iupac_high-confidence_v1_0.csv"
)
HERE = Path(__file__).resolve().parent
OUT = HERE / "data"

# The dataset grades every measurement. The weakest grade is dropped; the rest is kept, since
# "Approximate" is most of the data and still experimental.
REJECTED_ASSESSMENTS = {"Very uncertain", "Unknown"}

# pKa shifts with temperature. Restricting to exactly 25 degC would halve the data, so a window
# around room temperature is used and anything outside it is dropped.
MIN_TEMP_C, MAX_TEMP_C = 15.0, 35.0

TARGETS = {
    # pK1: the first acid dissociation — the molecule gives up a proton.
    "pka_acidic": "pK1",
    # pKAH1: the first protonation of a base — written as the pKa of its conjugate acid.
    "pka_basic": "pKAH1",
}


def temperature_ok(raw: str) -> bool:
    """Keep rows near room temperature, and rows that never stated one."""
    text = raw.strip()
    if not text or text.startswith("not_st"):
        return True
    # Values look like "25", "22.5+2.5", "20-25"; the leading number is enough to judge.
    number = ""
    for ch in text:
        if ch.isdigit() or (ch == "." and "." not in number):
            number += ch
        else:
            break
    if not number:
        return True
    try:
        return MIN_TEMP_C <= float(number) <= MAX_TEMP_C
    except ValueError:
        return True


def build(rows: list[dict], name: str, pka_type: str) -> None:
    by_smiles: dict[str, list[float]] = {}
    for row in rows:
        if row["pka_type"] != pka_type:
            continue
        if row["assessment"] in REJECTED_ASSESSMENTS:
            continue
        if not temperature_ok(row["T"]):
            continue
        try:
            value = float(row["pka_value"])
        except ValueError:
            continue
        try:
            smiles = chem.canonical_smiles(row["SMILES"])
        except Exception:
            continue
        by_smiles.setdefault(smiles, []).append(value)

    # Several measurements per compound is normal here, from different methods and references.
    # The median shrugs off a bad digitisation in a way the mean does not.
    merged = sorted((s, statistics.median(v)) for s, v in by_smiles.items())
    repeated = sum(1 for v in by_smiles.values() if len(v) > 1)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "target"])
        writer.writerows(merged)

    values = [v for _, v in merged]
    print(
        f"{name:12} ({pka_type:6}) {len(merged):5d} molecules  "
        f"({repeated} with repeat measurements)  "
        f"min {min(values):.2f} max {max(values):.2f} median {statistics.median(values):.2f}"
    )


def main() -> None:
    print(__doc__.split("Why two targets")[0].strip())
    print()
    print(f"downloading {SOURCE} ...")
    with urllib.request.urlopen(SOURCE) as response:
        text = response.read().decode("utf-8", errors="replace")
    rows = list(csv.DictReader(io.StringIO(text)))
    print(f"{len(rows)} measurements read")
    for name, pka_type in TARGETS.items():
        build(rows, name, pka_type)


if __name__ == "__main__":
    main()
