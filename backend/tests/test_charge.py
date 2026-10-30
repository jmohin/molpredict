"""The charge arithmetic, on inputs whose answers are known without a model."""

import math

import pytest

from molpredict_backend import charge
from molpredict_backend.charge import Site


def test_acid_is_half_deprotonated_at_its_pka():
    acid = Site(0, "acid", 4.76)
    assert charge.net_charge([acid], 4.76) == pytest.approx(-0.5)


def test_base_is_half_protonated_at_its_pka():
    base = Site(0, "base", 9.5)
    assert charge.net_charge([base], 9.5) == pytest.approx(0.5)


def test_acid_limits():
    """Far below its pKa an acid is neutral; far above it carries a full negative charge."""
    acid = Site(0, "acid", 7.0)
    assert charge.net_charge([acid], 0.0) == pytest.approx(0.0, abs=1e-6)
    assert charge.net_charge([acid], 14.0) == pytest.approx(-1.0, abs=1e-6)


def test_base_limits():
    base = Site(0, "base", 7.0)
    assert charge.net_charge([base], 0.0) == pytest.approx(1.0, abs=1e-6)
    assert charge.net_charge([base], 14.0) == pytest.approx(0.0, abs=1e-6)


def test_one_ph_unit_either_side_is_ten_to_one():
    """The defining property of the relationship, so worth asserting directly."""
    acid = Site(0, "acid", 5.0)
    assert charge.net_charge([acid], 6.0) == pytest.approx(-10 / 11)
    assert charge.net_charge([acid], 4.0) == pytest.approx(-1 / 11)


def test_zwitterion_is_neutral_between_its_pkas():
    """Glycine: carboxyl near 2.3, amine near 9.6. Between them it is +1 and -1 at once."""
    sites = [Site(0, "acid", 2.34), Site(1, "base", 9.60)]
    assert charge.net_charge(sites, 6.0) == pytest.approx(0.0, abs=0.01)
    assert charge.net_charge(sites, 0.0) == pytest.approx(1.0, abs=0.01)
    assert charge.net_charge(sites, 14.0) == pytest.approx(-1.0, abs=0.01)


def test_isoelectric_point_of_a_zwitterion_is_the_mean_of_its_pkas():
    sites = [Site(0, "acid", 2.34), Site(1, "base", 9.60)]
    assert charge.isoelectric_point(sites) == pytest.approx((2.34 + 9.60) / 2, abs=0.01)


def test_isoelectric_point_is_none_when_the_curve_never_crosses_zero():
    """A lone carboxylic acid is neutral or negative; it is never positive."""
    assert charge.isoelectric_point([Site(0, "acid", 4.76)]) is None


def test_permanent_cation_never_discharges():
    """A quaternary ammonium has no ionisable site and keeps its charge at every pH."""
    phs, charges = charge.curve([], formal_charge=1.0)
    assert set(charges) == {1.0}
    assert charge.isoelectric_point([], formal_charge=1.0) is None


def test_a_molecule_with_no_ionisable_site_has_no_isoelectric_point():
    """Hexane is neutral at every pH. Its curve crosses zero everywhere, which is not a pI."""
    assert charge.isoelectric_point([]) is None


def test_extreme_pka_does_not_overflow():
    """Predicted pKa values can land far outside the window; 10**400 would raise."""
    assert charge.net_charge([Site(0, "acid", 400.0)], 7.0) == pytest.approx(0.0)
    assert charge.net_charge([Site(0, "base", -400.0)], 7.0) == pytest.approx(0.0)


def test_curve_spans_the_window_exactly():
    phs, charges = charge.curve([Site(0, "acid", 7.0)])
    assert len(phs) == len(charges) == 141
    assert phs[0] == pytest.approx(0.0)
    # Accumulated 0.1 additions would land near 14.000000000000002 without integer stepping.
    assert phs[-1] == pytest.approx(14.0)
    assert math.isclose(phs[-1], 14.0, abs_tol=1e-12)


def test_curve_decreases_monotonically():
    """More alkaline is never more positive, which is what lets the crossing be bisected."""
    sites = [Site(0, "acid", 3.0), Site(1, "base", 8.0), Site(2, "acid", 11.0)]
    _, charges = charge.curve(sites)
    assert all(b <= a + 1e-12 for a, b in zip(charges, charges[1:]))


def test_relevant_drops_only_unreachable_sites():
    near = Site(0, "acid", 15.0)
    far = Site(1, "acid", 40.0)
    assert charge.relevant([near, far]) == [near]


def test_dominant_charge_rounds_to_a_whole_species():
    sites = [Site(0, "base", 9.5)]
    assert charge.dominant_charge(sites, 5.0) == 1
    assert charge.dominant_charge(sites, 13.0) == 0
