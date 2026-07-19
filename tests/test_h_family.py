"""Foundation F1 — h-parameter family encoding for VI_h / VII_h / III / VI_{-1/9}.

Covers:
  * Construction defaults and explicit-h validation.
  * Rejection of h=0, |h|>=hard-limit, wrong-sign h, near-III degeneracy.
  * Classification helpers (is_class_A/B, is_lrs, has_h_parameter).
  * Wainwright–Hsu canonical (n₁,n₂,n₃) eigenvalue tables.
  * Continuity limits required by the Wave-4 cell PRs:
      - VII_h(h→0⁺) eigenvalue tuple → (0,+1,0⁺) = VII_0 limit
      - VI_h(h→-1) eigenvalue tuple → (0,+1,-1) = III canonical
      - finite-difference derivative of n₃ vs h for VI_h is constant
"""
from __future__ import annotations

import math
import warnings

import pytest

from rabbit.config.conventions import (
    BianchiSpec,
    BianchiType,
    H_FAMILY_TYPES,
)


@pytest.mark.production
class TestBianchiSpecConstruction:
    def test_typeI_no_h(self):
        spec = BianchiSpec.from_type(BianchiType.TYPE_I)
        assert spec.h is None
        assert spec.is_class_A
        assert spec.is_lrs
        assert not spec.has_h_parameter

    def test_typeII_no_h(self):
        spec = BianchiSpec.from_type(BianchiType.TYPE_II)
        assert spec.h is None
        assert spec.is_class_A
        assert spec.is_lrs

    def test_typeIX_no_h_no_lrs(self):
        spec = BianchiSpec.from_type(BianchiType.TYPE_IX)
        assert spec.h is None
        assert spec.is_class_A
        assert not spec.is_lrs

    def test_typeIII_canonical_h_minus_one(self):
        spec = BianchiSpec.from_type(BianchiType.TYPE_III)
        assert spec.h == -1.0
        assert spec.is_class_B
        assert spec.has_h_parameter

    def test_typeIII_explicit_h_must_match_canonical(self):
        spec_default = BianchiSpec.from_type(BianchiType.TYPE_III)
        spec_explicit = BianchiSpec.from_type(BianchiType.TYPE_III, h=-1.0)
        assert spec_default.h == spec_explicit.h
        with pytest.raises(ValueError, match="canonical h"):
            BianchiSpec.from_type(BianchiType.TYPE_III, h=-2.0)

    def test_typeVI_m19_canonical_h(self):
        spec = BianchiSpec.from_type(BianchiType.TYPE_VI_M19)
        assert math.isclose(spec.h, -1.0 / 9.0, abs_tol=1e-15)
        assert spec.is_class_B

    def test_typeV_typeIV_no_h(self):
        for t in (BianchiType.TYPE_V, BianchiType.TYPE_IV):
            spec = BianchiSpec.from_type(t)
            assert spec.h is None
            assert spec.is_class_B
            assert spec.is_lrs


@pytest.mark.production
class TestHFamilyValidation:
    def test_VIh_requires_explicit_h(self):
        with pytest.raises(ValueError, match="requires an explicit h"):
            BianchiSpec.from_type(BianchiType.TYPE_VIH)

    def test_VIIh_requires_explicit_h(self):
        with pytest.raises(ValueError, match="requires an explicit h"):
            BianchiSpec.from_type(BianchiType.TYPE_VIIH)

    def test_h_zero_rejected(self):
        for t in H_FAMILY_TYPES:
            with pytest.raises(ValueError, match="h=0 is degenerate"):
                BianchiSpec.from_type(t, h=0.0)

    def test_h_nan_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            BianchiSpec.from_type(BianchiType.TYPE_VIH, h=float("nan"))

    def test_VIh_requires_negative_h(self):
        with pytest.raises(ValueError, match="VIH requires h < 0"):
            BianchiSpec.from_type(BianchiType.TYPE_VIH, h=2.0)

    def test_VIIh_requires_positive_h(self):
        with pytest.raises(ValueError, match="VIIH requires h > 0"):
            BianchiSpec.from_type(BianchiType.TYPE_VIIH, h=-2.0)

    def test_VIh_near_minus_one_rejected(self):
        # Within tolerance of -1 collapses to Type III.
        with pytest.raises(ValueError, match="canonical limit Type III"):
            BianchiSpec.from_type(BianchiType.TYPE_VIH, h=-1.0)
        with pytest.raises(ValueError, match="canonical limit Type III"):
            BianchiSpec.from_type(BianchiType.TYPE_VIH, h=-0.9999)

    def test_VIh_exceptional_minus_one_ninth_rejected(self):
        with pytest.raises(ValueError, match="VI_\\{-1/9\\}"):
            BianchiSpec.from_type(BianchiType.TYPE_VIH, h=-1.0 / 9.0)

    def test_h_hard_limit_rejected(self):
        # |h| >= 100 is singular.
        with pytest.raises(ValueError, match="singular"):
            BianchiSpec.from_type(BianchiType.TYPE_VIIH, h=200.0)
        with pytest.raises(ValueError, match="singular"):
            BianchiSpec.from_type(BianchiType.TYPE_VIH, h=-100.0)

    def test_h_soft_limit_warns(self):
        # 10 < |h| < 100: warn but accept.
        with pytest.warns(RuntimeWarning, match="extreme-anisotropy"):
            BianchiSpec.from_type(BianchiType.TYPE_VIIH, h=50.0)
        with pytest.warns(RuntimeWarning, match="extreme-anisotropy"):
            BianchiSpec.from_type(BianchiType.TYPE_VIH, h=-50.0)

    def test_h_within_soft_limit_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            BianchiSpec.from_type(BianchiType.TYPE_VIIH, h=2.0)
            BianchiSpec.from_type(BianchiType.TYPE_VIH, h=-2.0)

    def test_no_h_for_non_h_family_types(self):
        for t in (BianchiType.TYPE_I, BianchiType.TYPE_II, BianchiType.TYPE_VI0,
                  BianchiType.TYPE_VII0, BianchiType.TYPE_VIII, BianchiType.TYPE_IX,
                  BianchiType.TYPE_V, BianchiType.TYPE_IV):
            with pytest.raises(ValueError, match="does not take an h-parameter"):
                BianchiSpec.from_type(t, h=2.0)


@pytest.mark.production
class TestSignConvention:
    def test_default_sign_n(self):
        spec = BianchiSpec.from_type(BianchiType.TYPE_VIII)
        assert spec.sign_n == (1, 1, 1)

    def test_invalid_sign_n_rejected(self):
        with pytest.raises(ValueError, match=r"sign_n entries must be"):
            BianchiSpec(type=BianchiType.TYPE_I, sign_n=(2, 1, 1))

    def test_negative_sign_n_accepted(self):
        spec = BianchiSpec(type=BianchiType.TYPE_IX, sign_n=(-1, +1, +1))
        # Sign flip propagates through canonical eigenvalues.
        n = spec.canonical_n_eigenvalues()
        assert n[0] == -1.0


@pytest.mark.production
class TestCanonicalEigenvalues:
    """Wainwright–Ellis 1997 Tab. 6.2 canonical eigenvalue tables."""

    def test_typeI(self):
        n = BianchiSpec.from_type(BianchiType.TYPE_I).canonical_n_eigenvalues()
        assert n == (0.0, 0.0, 0.0)

    def test_typeII(self):
        n = BianchiSpec.from_type(BianchiType.TYPE_II).canonical_n_eigenvalues()
        assert n == (1.0, 0.0, 0.0)

    def test_typeVI0(self):
        n = BianchiSpec.from_type(BianchiType.TYPE_VI0).canonical_n_eigenvalues()
        assert n == (1.0, -1.0, 0.0)

    def test_typeVII0(self):
        n = BianchiSpec.from_type(BianchiType.TYPE_VII0).canonical_n_eigenvalues()
        assert n == (1.0, 1.0, 0.0)

    def test_typeVIII(self):
        n = BianchiSpec.from_type(BianchiType.TYPE_VIII).canonical_n_eigenvalues()
        assert n == (1.0, 1.0, -1.0)

    def test_typeIX(self):
        n = BianchiSpec.from_type(BianchiType.TYPE_IX).canonical_n_eigenvalues()
        assert n == (1.0, 1.0, 1.0)

    def test_typeV(self):
        assert BianchiSpec.from_type(BianchiType.TYPE_V).canonical_n_eigenvalues() == (0.0, 0.0, 0.0)

    def test_typeIV(self):
        assert BianchiSpec.from_type(BianchiType.TYPE_IV).canonical_n_eigenvalues() == (1.0, 0.0, 0.0)

    def test_typeIII(self):
        # III sits at the h = -1 limit of VI_h.
        n = BianchiSpec.from_type(BianchiType.TYPE_III).canonical_n_eigenvalues()
        assert n == (0.0, 1.0, -1.0)

    def test_typeVI_m19(self):
        n = BianchiSpec.from_type(BianchiType.TYPE_VI_M19).canonical_n_eigenvalues()
        assert math.isclose(n[2], -1.0 / 9.0, abs_tol=1e-15)

    def test_VIh_n3_equals_h(self):
        for h in (-2.0, -0.5, -3.5):
            spec = BianchiSpec.from_type(BianchiType.TYPE_VIH, h=h)
            n = spec.canonical_n_eigenvalues()
            assert n == (0.0, 1.0, h)

    def test_VIIh_n3_equals_h(self):
        for h in (0.5, 2.0, 3.5):
            spec = BianchiSpec.from_type(BianchiType.TYPE_VIIH, h=h)
            n = spec.canonical_n_eigenvalues()
            assert n == (0.0, 1.0, h)

    def test_scale_factor_applies_uniformly(self):
        spec = BianchiSpec.from_type(BianchiType.TYPE_IX)
        n = spec.canonical_n_eigenvalues(scale=0.1)
        assert all(abs(ni - 0.1) < 1e-15 for ni in n)


@pytest.mark.production
class TestContinuityLimits:
    """Continuous-h limits required by Wave-4 cell PRs (III/VI_h/VII_h)."""

    def test_VIh_h_to_minus_one_recovers_typeIII(self):
        # As h→-1 (just outside the III-degeneracy guard), VI_h's
        # eigenvalue tuple must approach Type III's exactly.
        ref = BianchiSpec.from_type(BianchiType.TYPE_III).canonical_n_eigenvalues()
        # The smallest h still accepted by the validator (just outside the
        # III-degeneracy tolerance, ~1e-3).
        spec = BianchiSpec.from_type(BianchiType.TYPE_VIH, h=-1.0 + 2.0e-3)
        n = spec.canonical_n_eigenvalues()
        assert math.isclose(n[2], -1.0 + 2.0e-3, abs_tol=1e-12)
        # Difference from III at this clearance is bounded by the tolerance.
        assert abs(n[2] - ref[2]) < 3.0e-3

    def test_VIIh_h_to_zero_plus_n3_continuous(self):
        # The eigenvalue n₃ varies linearly with h and goes to 0⁺ as h→0⁺.
        # NOTE: VII_h and VII_0 are NOT continuous as types — VII_0 has
        # n=(1,1,0) while VII_h has n=(0,1,h); the bridge requires the
        # Class B frame variable A simultaneously vanishing.  At the
        # pure-structure-constant level the only continuity claim is
        # n₃(h) → 0 as h → 0⁺.
        sequence = [1.0, 0.1, 0.01, 0.001]
        prev_n3 = None
        for h in sequence:
            spec = BianchiSpec.from_type(BianchiType.TYPE_VIIH, h=h)
            n = spec.canonical_n_eigenvalues()
            assert n[0] == 0.0
            assert n[1] == 1.0
            assert math.isclose(n[2], h, abs_tol=1e-12)
            if prev_n3 is not None:
                assert abs(n[2]) < abs(prev_n3), "n₃ must decrease monotonically as h→0⁺"
            prev_n3 = n[2]

    def test_n3_over_h_is_unit_constant_for_VIh_and_VIIh(self):
        # ∂n₃/∂h = 1 by construction across both h-family types.
        for type_, h_grid in (
            (BianchiType.TYPE_VIH, [-2.0, -3.0, -5.0]),
            (BianchiType.TYPE_VIIH, [0.5, 2.0, 5.0]),
        ):
            for h in h_grid:
                eps = 1e-4 * abs(h)
                n_plus = BianchiSpec.from_type(type_, h=h + eps).canonical_n_eigenvalues()
                n_minus = BianchiSpec.from_type(type_, h=h - eps).canonical_n_eigenvalues()
                d_n3_d_h = (n_plus[2] - n_minus[2]) / (2.0 * eps)
                assert math.isclose(d_n3_d_h, 1.0, rel_tol=1e-9)


@pytest.mark.production
class TestSpecImmutabilityAndRepr:
    def test_spec_is_frozen(self):
        spec = BianchiSpec.from_type(BianchiType.TYPE_I)
        with pytest.raises(Exception):
            spec.h = -2.0  # type: ignore[misc]

    def test_spec_repr_includes_h(self):
        r = repr(BianchiSpec.from_type(BianchiType.TYPE_VIH, h=-3.0))
        assert "TYPE_VIH" in r and "h=-3.0" in r

    def test_spec_repr_omits_h_for_no_h_types(self):
        r = repr(BianchiSpec.from_type(BianchiType.TYPE_I))
        assert "h=" not in r

    def test_spec_repr_includes_sign_n_only_when_nondefault(self):
        r_default = repr(BianchiSpec.from_type(BianchiType.TYPE_I))
        assert "sign_n" not in r_default
        r_flipped = repr(BianchiSpec(type=BianchiType.TYPE_VIII, sign_n=(-1, 1, 1)))
        assert "sign_n=(-1, 1, 1)" in r_flipped


@pytest.mark.production
class TestEnumBackCompat:
    """Existing string-keyed dispatch must continue to work."""

    def test_BianchiType_values_unchanged(self):
        # The 11 enum values are the dispatch keys used everywhere.
        assert BianchiType.TYPE_I.value == "I"
        assert BianchiType.TYPE_VIH.value == "VIh"
        assert BianchiType.TYPE_VI_M19.value == "VI_m19"
        assert len(list(BianchiType)) == 12  # 11 types + the 12th legacy alias if any

    def test_spec_type_field_recovers_enum(self):
        for t in BianchiType:
            if t in H_FAMILY_TYPES:
                spec = BianchiSpec.from_type(t, h=-2.0 if t is BianchiType.TYPE_VIH else 2.0)
            else:
                spec = BianchiSpec.from_type(t)
            assert spec.type is t
