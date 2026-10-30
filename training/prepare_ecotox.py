"""Turn EPA's ECOTOX ASCII release into training CSVs for two aquatic LC50 endpoints.

    uv run python prepare_ecotox.py          # downloads the archive if absent

Writes data/fathead_minnow_lc50.csv and data/daphnia_magna_lc50.csv, two columns each.

ECOTOX carries no structures — only CAS numbers and DSSTox IDs — so SMILES come from OPERA's
archive, which is already downloaded for the other endpoints and whose CATMoS, CERAPP and
CoMPARA files carry CASRN and DTXSID beside QSAR-ready structures. That keeps the structure
source identical to the rest of this project rather than introducing a second convention.

ECOTOX is a bibliographic abstraction of thousands of separate studies, which means it holds
several kinds of row that look numeric but are not comparable:

  * Censored values. A result of ">100 mg/L" says the LC50 was never reached. Treating 100 as
    the answer teaches the model a value the experiment explicitly did not measure, and these
    cluster at round numbers, so the damage is systematic rather than noise. The operator lives
    in its OWN column, `conc1_mean_op`, and `conc1_mean` holds a bare number either way — so a
    filter that inspects the value cannot see the censoring and keeps every one of these.
  * Mixed units. Concentrations appear as mg/L, ug/L, ppm, ppb, nM and dozens of others,
    including units of a dosed carrier rather than the chemical. Only mass- and mole-per-volume
    units of the test substance itself are usable.
  * Mixed durations. A 24-hour and a 96-hour LC50 for the same chemical are different numbers.
    The standard test is 96 h for fathead minnow and 48 h for Daphnia magna, and those are the
    only durations kept.
  * Non-mortality effects. The endpoint column holds LC50 for lethality but the effect column
    distinguishes mortality from growth, reproduction and behaviour.

The target is pLC50, -log10(mol/L): the convention in the aquatic-toxicity literature, so the
numbers are comparable with published models, and higher means more toxic.
"""

import csv
import io
import json
import math
import statistics
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))
from molpredict_backend import chem  # noqa: E402

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
CACHE = OUT / "ecotox_ascii.zip"
OPERA = OUT / "OPERA_Data.zip"
# The EPA publishes one dated archive at a time and the landing page links to the current one.
ECOTOX_URL = "https://gaftp.epa.gov/ecotox/ecotox_ascii_09_15_2026.zip"

# Concentration units we can convert to mol/L, as multipliers to g/L. ppm and ppb are taken as
# mg/L and ug/L: in freshwater at test concentrations the difference from density is far below
# the spread between laboratories.
TO_GRAMS_PER_L = {
    "g/L": 1.0,
    "mg/L": 1e-3,
    "ug/L": 1e-6,
    "ng/L": 1e-9,
    "pg/L": 1e-12,
    "ppm": 1e-3,  # in dilute freshwater, 1 ppm = 1 mg/L
    "ppb": 1e-6,
    "mg/dm3": 1e-3,
    "ug/dm3": 1e-6,
    "g/ml": 1e3,
    "mg/ml": 1.0,
    "ug/ml": 1e-3,
    "ng/ml": 1e-6,
}
# "ppt" is deliberately absent: it means parts per trillion in some studies and parts per
# thousand in others, a factor of a billion apart, and the table does not say which.

# Molar units, as multipliers to mol/L.
TO_MOLES_PER_L = {
    "M": 1.0,
    "mM": 1e-3,
    "uM": 1e-6,
    "nM": 1e-9,
    "pM": 1e-12,
    "mol/L": 1.0,
    "mmol/L": 1e-3,
    "umol/L": 1e-6,
    "nmol/L": 1e-9,
}

# `conc1_mean_op` holds the comparison operator when the LC50 was bounded rather than measured.
CENSORING_OPERATORS = {">", "<", ">=", "<=", "~"}

# Freshwater species in fresh or plain aqueous media. Salt water, sediment and the soil media
# are a different experiment; unspecified is kept because the species itself settles it.
MEDIA_ALLOWED = {"FW", "AQU", "", "--", "NR", "NC", "UKN"}
# Laboratory only. A field LC50 is confounded by everything else in the water.
LOCATION_ALLOWED = {"LAB", "", "--", "NR", "NC"}
# "F" is the concentration of a formulated product, not of the chemical the SMILES describes.
CONCENTRATION_TYPE_REJECTED = {"F"}


@dataclass(frozen=True)
class Endpoint:
    name: str
    latin_name: str
    duration_hours: float


ENDPOINTS = [
    Endpoint("fathead_minnow_lc50", "Pimephales promelas", 96.0),
    Endpoint("daphnia_magna_lc50", "Daphnia magna", 48.0),
]

# Effect codes for mortality. ECOTOX uses MOR, sometimes with a qualifier appended.
MORTALITY_PREFIX = "MOR"
ENDPOINT_WANTED = "LC50"


def fetch(attempts: int = 10) -> Path:
    """Download the archive, resuming if the connection drops.

    The EPA's host closed the connection partway through every attempt made while writing this,
    at a few hundred megabytes. A plain urlretrieve leaves a truncated file behind that still
    looks like a zip until something reads the central directory, so this resumes by byte range
    and only moves the file into place once the archive opens.
    """
    if CACHE.exists():
        return CACHE
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    partial = CACHE.with_suffix(".part")
    print(f"downloading {ECOTOX_URL} ...")
    for attempt in range(attempts):
        have = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(ECOTOX_URL)
        if have:
            request.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(request) as response, partial.open("ab") as out:
                # A server that ignores the Range header answers 200 and resends from zero,
                # which would corrupt what we already have.
                if have and response.status != 206:
                    out.close()
                    partial.unlink()
                    continue
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            break
        except Exception as error:  # noqa: BLE001 — any transport failure is retried
            print(f"  attempt {attempt + 1} stopped at {partial.stat().st_size:,} bytes: {error}")
    try:
        with zipfile.ZipFile(partial) as zf:
            if zf.testzip() is not None:
                raise zipfile.BadZipFile("a member failed its CRC check")
    except zipfile.BadZipFile as error:
        raise SystemExit(f"download incomplete after {attempts} attempts: {error}") from error
    partial.rename(CACHE)
    return CACHE


def member(zf: zipfile.ZipFile, stem: str) -> str:
    """Find one table by filename. `species` and `chemicals` live under `validation/`."""
    hits = [n for n in zf.namelist() if n.rsplit("/", 1)[-1].lower() == f"{stem}.txt"]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one {stem}.txt in the archive, found {hits}")
    return hits[0]


def rows(zf: zipfile.ZipFile, name: str):
    with zf.open(name) as handle:
        text = io.TextIOWrapper(handle, encoding="utf-8", errors="replace")
        yield from csv.DictReader(text, delimiter="|")


def clean(value: str) -> str:
    return (value or "").strip()


def duration_hours(value: str, unit: str) -> float | None:
    """ECOTOX records duration in whatever unit the study used."""
    try:
        amount = float(value)
    except ValueError:
        return None
    factor = {"h": 1.0, "d": 24.0, "mi": 1 / 60.0, "wk": 168.0}.get(unit.strip().lower())
    return amount * factor if factor else None


def concentration_mol_per_l(raw: str, unit: str, mol_weight: float) -> float | None:
    raw = clean(raw)
    if not raw:
        return None
    try:
        amount = float(raw)
    except ValueError:
        return None
    if amount <= 0:
        return None
    unit = clean(unit)
    if unit in TO_MOLES_PER_L:
        return amount * TO_MOLES_PER_L[unit]
    if unit in TO_GRAMS_PER_L:
        return amount * TO_GRAMS_PER_L[unit] / mol_weight
    return None


def load_structures() -> dict[str, str]:
    """CAS and DTXSID to SMILES, from OPERA's archive."""
    if not OPERA.exists():
        raise SystemExit(f"{OPERA} is missing — run prepare_data.py first")
    lookup: dict[str, str] = {}
    with zipfile.ZipFile(OPERA) as zf:
        for name in [n for n in zf.namelist() if n.endswith(".sdf")]:
            with zf.open(name) as handle:
                for mol in Chem.ForwardSDMolSupplier(handle, sanitize=False):
                    if mol is None:
                        continue
                    smiles = ""
                    for tag in ("Canonical_QSARr", "Original_SMILES"):
                        if mol.HasProp(tag) and clean(mol.GetProp(tag)):
                            smiles = clean(mol.GetProp(tag))
                            break
                    if not smiles:
                        continue
                    for tag in ("CASRN", "casrn", "DTXSID"):
                        if not mol.HasProp(tag):
                            continue
                        for key in mol.GetProp(tag).replace("|", " ").split():
                            lookup.setdefault(normalise_cas(key), smiles)
    return lookup


def normalise_cas(value: str) -> str:
    """ECOTOX stores CAS as digits only; OPERA writes them hyphenated."""
    value = clean(value)
    return value.replace("-", "") if value[:1].isdigit() else value


def main() -> None:
    archive = fetch()
    print("reading OPERA structures ...")
    structures = load_structures()
    print(f"  {len(structures)} CAS/DTXSID keys")

    with zipfile.ZipFile(archive) as zf:
        # ECOTOX's own chemical table maps CAS to a DSSTox id, which reaches a second set of
        # OPERA structures for chemicals whose CAS is not itself in those files.
        recovered = 0
        for row in rows(zf, member(zf, "chemicals")):
            cas = normalise_cas(clean(row.get("cas_number", "")))
            dtxsid = clean(row.get("dtxsid", ""))
            if cas and dtxsid and cas not in structures and dtxsid in structures:
                structures[cas] = structures[dtxsid]
                recovered += 1
        print(f"  {recovered} further CAS resolved through ECOTOX's DSSTox ids")

        wanted_species = {e.latin_name.lower(): e for e in ENDPOINTS}
        species_ids: dict[str, Endpoint] = {}
        for row in rows(zf, member(zf, "species")):
            endpoint = wanted_species.get(clean(row.get("latin_name", "")).lower())
            if endpoint:
                species_ids[clean(row["species_number"])] = endpoint
        print(f"  species numbers: { {e.name: n for n, e in species_ids.items()} }")

        # test_id -> (endpoint, cas), for laboratory freshwater tests on those species.
        tests: dict[str, tuple[Endpoint, str]] = {}
        dropped_tests: dict[str, int] = {}
        for row in rows(zf, member(zf, "tests")):
            endpoint = species_ids.get(clean(row.get("species_number", "")))
            if not endpoint:
                continue
            if clean(row.get("test_location", "")).upper() not in LOCATION_ALLOWED:
                dropped_tests["not a laboratory test"] = (
                    dropped_tests.get("not a laboratory test", 0) + 1
                )
                continue
            if clean(row.get("media_type", "")).upper() not in MEDIA_ALLOWED:
                dropped_tests["media is not fresh water"] = (
                    dropped_tests.get("media is not fresh water", 0) + 1
                )
                continue
            tests[clean(row["test_id"])] = (endpoint, clean(row.get("test_cas", "")))
        print(f"  {len(tests)} usable tests on those species")
        for reason, count in sorted(dropped_tests.items(), key=lambda kv: -kv[1]):
            print(f"    dropped {count} tests: {reason}")

        reasons: dict[str, int] = {}
        by_endpoint: dict[str, dict[str, list[float]]] = {e.name: {} for e in ENDPOINTS}
        weights: dict[str, float] = {}

        def note(reason: str) -> None:
            reasons[reason] = reasons.get(reason, 0) + 1

        for row in rows(zf, member(zf, "results")):
            found = tests.get(clean(row.get("test_id", "")))
            if not found:
                continue
            endpoint, cas = found
            if clean(row.get("endpoint", "")).rstrip("*/") != ENDPOINT_WANTED:
                note("endpoint is not LC50")
                continue
            if not clean(row.get("effect", "")).upper().startswith(MORTALITY_PREFIX):
                note("effect is not mortality")
                continue
            if clean(row.get("conc1_mean_op", "")) in CENSORING_OPERATORS:
                note("concentration is censored (a bound, not a measurement)")
                continue
            if clean(row.get("conc1_type", "")).upper() in CONCENTRATION_TYPE_REJECTED:
                note("concentration is of a formulated product")
                continue
            if clean(row.get("obs_duration_mean_op", "")) in CENSORING_OPERATORS:
                note("duration is censored")
                continue
            hours = duration_hours(
                clean(row.get("obs_duration_mean", "")), clean(row.get("obs_duration_unit", ""))
            )
            if hours is None:
                note("duration unreadable")
                continue
            if abs(hours - endpoint.duration_hours) > 0.01:
                note("duration is not the standard test length")
                continue
            smiles_raw = structures.get(normalise_cas(cas))
            if not smiles_raw:
                note("no structure for this CAS")
                continue
            if smiles_raw not in weights:
                mol = Chem.MolFromSmiles(smiles_raw)
                weights[smiles_raw] = Descriptors.MolWt(mol) if mol else 0.0
            mol_weight = weights[smiles_raw]
            if not mol_weight:
                note("structure unparseable")
                continue
            molar = concentration_mol_per_l(
                row.get("conc1_mean", ""), row.get("conc1_unit", ""), mol_weight
            )
            if molar is None:
                note("concentration censored, absent or in an unusable unit")
                continue
            try:
                smiles = chem.canonical_smiles(smiles_raw)
            except Exception:
                note("structure rejected by standardisation")
                continue
            by_endpoint[endpoint.name].setdefault(smiles, []).append(-math.log10(molar))

    print("\nrows dropped:")
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {count:8d}  {reason}")

    print()
    OUT.mkdir(parents=True, exist_ok=True)
    for endpoint in ENDPOINTS:
        measurements = by_endpoint[endpoint.name]
        # Median, not mean: replicate LC50s for one chemical span laboratories and decades, and
        # a single mistyped order of magnitude would drag a mean with it.
        rows_out = sorted((s, statistics.median(v)) for s, v in measurements.items())
        path = OUT / f"{endpoint.name}.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles", "target"])
            writer.writerows(rows_out)
        values = [r[1] for r in rows_out]
        repeats = sum(1 for v in measurements.values() if len(v) > 1)
        print(
            f"{endpoint.name:22} {len(rows_out):5d} molecules  "
            f"({sum(len(v) for v in measurements.values())} measurements, {repeats} with repeats)  "
            f"min {min(values):.2f} max {max(values):.2f} mean {statistics.fmean(values):.2f}  "
            f"[pLC50, -log10(mol/L)]"
        )


if __name__ == "__main__":
    main()
