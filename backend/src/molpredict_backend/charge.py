"""Net charge as a function of pH, from per-site pKa values.

Each ionisable site is treated as independent, which is the Henderson-Hasselbalch
approximation. An acidic site with pKa a carries charge -1 once deprotonated, so it
contributes

    -1 / (1 + 10**(a - pH))

and a basic site whose conjugate acid has pKa b carries +1 while protonated, contributing

    +1 / (1 + 10**(pH - b))

Summing those over every site gives the curve. This is NOT a full microspecies equilibrium:
real sites interact, so deprotonating one shifts its neighbour's pKa, and a dication's two
amine pKa values are not independent. Commercial tools solve the coupled equilibrium instead.
The independent-site curve has the right shape, the right limits and transitions in the right
places, and it is what the per-site inputs support; the error it introduces is largest for
molecules whose ionisable groups sit close together.

Kept free of any model so the arithmetic can be tested on hand-checkable inputs.
"""

import math
from dataclasses import dataclass

# Below 0 and above 14 is outside what an aqueous measurement means.
PH_MIN = 0.0
PH_MAX = 14.0
PH_STEP = 0.1
PHYSIOLOGICAL_PH = 7.4

# A site this far outside the window can never bend the curve inside it: 10**-6 of a charge
# unit. Carrying them would draw sites the plot cannot explain.
RELEVANCE_MARGIN = 6.0


@dataclass(frozen=True)
class Site:
    """One ionisable centre and its predicted pKa."""

    atom_index: int
    kind: str  # "acid" or "base"
    pka: float


def _occupancy(site: Site, ph: float) -> float:
    """Signed charge this site carries at this pH, between -1 and +1."""
    # exponent is clamped because 10**400 overflows while the limit it stands for is just 0 or 1.
    if site.kind == "acid":
        exponent = min(max(site.pka - ph, -30.0), 30.0)
        return -1.0 / (1.0 + 10.0**exponent)
    exponent = min(max(ph - site.pka, -30.0), 30.0)
    return 1.0 / (1.0 + 10.0**exponent)


def net_charge(sites: list[Site], ph: float, formal_charge: float = 0.0) -> float:
    """Net charge at one pH.

    `formal_charge` is the charge of the neutral reference species, which is zero for anything
    that has been neutralised but not for a permanent cation such as a quaternary ammonium.
    """
    return formal_charge + sum(_occupancy(s, ph) for s in sites)


def relevant(sites: list[Site]) -> list[Site]:
    """Drop sites whose pKa is so far outside 0-14 that they cannot affect the curve."""
    return [s for s in sites if PH_MIN - RELEVANCE_MARGIN <= s.pka <= PH_MAX + RELEVANCE_MARGIN]


def curve(
    sites: list[Site], formal_charge: float = 0.0
) -> tuple[list[float], list[float]]:
    """The curve over the pH window, as (pH values, charges)."""
    steps = round((PH_MAX - PH_MIN) / PH_STEP)
    # Built from an integer index so floating-point addition cannot drift the last point off 14.
    phs = [PH_MIN + i * PH_STEP for i in range(steps + 1)]
    return phs, [net_charge(sites, ph, formal_charge) for ph in phs]


def isoelectric_point(sites: list[Site], formal_charge: float = 0.0) -> float | None:
    """The pH where net charge crosses zero, or None if it never does in 0-14.

    Bisection on a monotonically decreasing function: every acidic site becomes more negative
    with pH and every basic site less positive, so the sum can cross zero at most once.
    """
    # A molecule with no ionisable site and no permanent charge is neutral at every pH. Its
    # curve crosses zero everywhere, which is not an isoelectric point — returning the window's
    # lower bound would put a confident "pI 0.0" on hexane.
    if not sites and formal_charge == 0.0:
        return None

    low, high = PH_MIN, PH_MAX
    low_charge = net_charge(sites, low, formal_charge)
    high_charge = net_charge(sites, high, formal_charge)
    if low_charge == 0.0:
        return low
    if high_charge == 0.0:
        return high
    if (low_charge > 0.0) == (high_charge > 0.0):
        return None
    for _ in range(60):
        mid = (low + high) / 2.0
        if (net_charge(sites, mid, formal_charge) > 0.0) == (low_charge > 0.0):
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def dominant_charge(sites: list[Site], ph: float, formal_charge: float = 0.0) -> int:
    """The nearest whole charge at this pH — the species a chemist would draw."""
    return int(math.floor(net_charge(sites, ph, formal_charge) + 0.5))
