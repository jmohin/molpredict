"""Turn EPA/NIEHS OPERA's released archive into one training CSV per endpoint.

    uv run python prepare_data.py            # downloads OPERA_Data.zip if absent

Writes data/<endpoint>.csv with two columns, `smiles` and `target`.

The SMILES are standardised the same way the API standardises a drawn molecule — largest
fragment, neutralised — so the model is trained on the form it will be asked about. Duplicates
that collapse to the same structure are averaged, since OPERA carries several measurements for
some chemicals.
"""

import csv
import math
import statistics
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))
from molpredict_backend import chem  # noqa: E402

RDLogger.DisableLog("rdApp.*")

OPERA_URL = "https://github.com/NIEHS/OPERA/raw/refs/heads/master/OPERA_Data.zip"
HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "OPERA_Data.zip"
OUT = HERE / "data"


@dataclass(frozen=True)
class Endpoint:
    name: str
    member: str  # file inside the zip
    tag: str  # SD property holding the value
    units: str
    task: str = "regression"
    log10: bool = False  # take log10 of the value first


ENDPOINTS = [
    Endpoint("boiling_point", "BP_QR.sdf", "BP", "degC"),
    Endpoint("melting_point", "MP_QR.sdf", "MP", "degC"),
    # Already log10(mmHg) in the file.
    Endpoint("vapor_pressure", "VP_QR.sdf", "LogVP", "log10(mmHg)"),
    Endpoint(
        "ready_biodegradability",
        "RBiodeg_QR.sdf",
        "Ready_Biodeg",
        "probability",
        task="classification",
    ),
    # Spans several orders of magnitude, so it is learned on a log scale, as OPERA's own
    # model is.
    Endpoint("rat_oral_ld50", "CATMoS_QR50k.sdf", "CATMoS_LD50_mgkg", "log10(mg/kg)", log10=True),
]


def fetch_archive() -> Path:
    if CACHE.exists():
        return CACHE
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {OPERA_URL} ...")
    urllib.request.urlretrieve(OPERA_URL, CACHE)
    return CACHE


def records(archive: Path, endpoint: Endpoint):
    """Yield (standardised smiles, value) from one SDF inside the archive."""
    with zipfile.ZipFile(archive) as zf, zf.open(endpoint.member) as handle:
        supplier = Chem.ForwardSDMolSupplier(handle, sanitize=True)
        for mol in supplier:
            if mol is None or not mol.GetNumAtoms():
                continue
            if not mol.HasProp(endpoint.tag):
                continue
            raw = mol.GetProp(endpoint.tag).strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if endpoint.log10:
                if value <= 0:
                    continue
                value = math.log10(value)
            if not math.isfinite(value):
                continue
            try:
                # The SDF holds the QSAR-ready structure; this puts it in exactly the form the
                # API will hand the model at prediction time.
                smiles = chem.canonical_smiles(Chem.MolToSmiles(mol))
            except Exception:
                continue
            yield smiles, value


def build(endpoint: Endpoint, archive: Path) -> Path:
    by_smiles: dict[str, list[float]] = {}
    for smiles, value in records(archive, endpoint):
        by_smiles.setdefault(smiles, []).append(value)

    collisions = sum(1 for v in by_smiles.values() if len(v) > 1)
    rows = []
    for smiles, values in by_smiles.items():
        value = statistics.fmean(values)
        if endpoint.task == "classification":
            # Averaging 0/1 labels can land between them; an even split is unusable.
            if value in (0.0, 1.0):
                rows.append((smiles, int(value)))
            continue
        rows.append((smiles, value))
    rows.sort()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{endpoint.name}.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "target"])
        writer.writerows(rows)

    values = [r[1] for r in rows]
    spread = (
        f"min {min(values):.2f} max {max(values):.2f} mean {statistics.fmean(values):.2f}"
        if endpoint.task == "regression"
        else f"positives {sum(values)}/{len(values)} ({100 * statistics.fmean(values):.0f}%)"
    )
    print(
        f"{endpoint.name:24} {len(rows):6d} molecules  "
        f"({collisions} merged duplicates)  {spread}  [{endpoint.units}]"
    )
    return path


def main() -> None:
    archive = fetch_archive()
    wanted = sys.argv[1:]
    for endpoint in ENDPOINTS:
        if wanted and endpoint.name not in wanted:
            continue
        build(endpoint, archive)


if __name__ == "__main__":
    main()
