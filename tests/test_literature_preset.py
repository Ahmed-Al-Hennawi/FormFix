"""
The literature-derived squat preset, which is offered for comparison rather
than adopted. These check it only moves values a paper actually speaks to,
leaves the defaults alone, and carries its provenance with it.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from exercises.squat.config import DEFAULT_CONFIG, rule_specs
from exercises.squat.literature_config import (
    LITERATURE_CONFIG,
    PROVENANCE,
    literature_config,
)

# Values a paper in the reviewed set actually speaks to.
TECHNIQUE_FIELDS = {
    "DEPTH_KNEE_ANGLE_PASS",
    "DEPTH_KNEE_ANGLE_WARN",
    "TORSO_LEAN_WARN",
    "TORSO_LEAN_FAIL",
    "HEEL_LIFT_THRESHOLD",
    "FULL_EXTENSION_TOLERANCE",
}


class TestItOnlyChangesWhatItCanCite:
    def test_no_engineering_threshold_moves(self):
        # No paper says anything about EMA weights or hysteresis levels, so
        # moving one under a citation borrows authority it doesn't have.
        for field in fields(DEFAULT_CONFIG):
            if field.name in TECHNIQUE_FIELDS or field.name == "notes":
                continue
            assert getattr(LITERATURE_CONFIG, field.name) == getattr(
                DEFAULT_CONFIG, field.name
            ), field.name

    def test_every_changed_value_has_a_recorded_reason(self):
        changed = {
            field.name
            for field in fields(DEFAULT_CONFIG)
            if field.name != "notes"
            and getattr(LITERATURE_CONFIG, field.name) != getattr(DEFAULT_CONFIG, field.name)
        }
        for name in changed:
            # A FAIL bar is covered by its paired WARN entry.
            root = name.replace("_FAIL", "_WARN")
            assert name in PROVENANCE or root in PROVENANCE, name

    def test_every_provenance_entry_names_its_source(self):
        for key, why in PROVENANCE.items():
            assert any(name in why for name in ("Kotiuk", "Dill", "Simoes")), key

    def test_the_preset_labels_itself_in_the_debug_notes(self):
        assert "LITERATURE" in LITERATURE_CONFIG.notes["preset"]
        assert "NOT validated" in LITERATURE_CONFIG.notes["preset"]

    def test_the_default_configuration_is_untouched_by_building_the_preset(self):
        before = DEFAULT_CONFIG.DEPTH_KNEE_ANGLE_PASS
        literature_config()
        assert before == DEFAULT_CONFIG.DEPTH_KNEE_ANGLE_PASS


class TestTheValuesThemselves:
    def test_it_demands_a_deeper_squat_than_the_operational_default(self):
        # Kotiuk et al. measured a supervised parallel squat, and a lower
        # interior knee angle means a deeper one.
        assert LITERATURE_CONFIG.DEPTH_KNEE_ANGLE_PASS < DEFAULT_CONFIG.DEPTH_KNEE_ANGLE_PASS

    def test_the_depth_bands_stay_ordered(self):
        assert LITERATURE_CONFIG.DEPTH_KNEE_ANGLE_PASS < LITERATURE_CONFIG.DEPTH_KNEE_ANGLE_WARN

    def test_the_depth_band_is_the_empirical_separation_dill_measured(self):
        gap = LITERATURE_CONFIG.DEPTH_KNEE_ANGLE_WARN - LITERATURE_CONFIG.DEPTH_KNEE_ANGLE_PASS
        assert gap == pytest.approx(20.0)

    @pytest.mark.parametrize(
        "rule_id, field",
        [("heel_lift", "HEEL_LIFT_THRESHOLD")],
    )
    def test_it_lifts_the_thresholds_that_sat_below_their_own_noise_floor(self, rule_id, field):
        from exercises.common.uncertainty import band_for

        band = band_for("", rule_id=rule_id)
        assert getattr(DEFAULT_CONFIG, field) < band.total, "the default should be below it"
        assert getattr(LITERATURE_CONFIG, field) > band.total, "the preset should be above it"

    def test_every_warning_bar_stays_below_its_failure_bar(self):
        assert LITERATURE_CONFIG.TORSO_LEAN_WARN < LITERATURE_CONFIG.TORSO_LEAN_FAIL


class TestItStillProducesAWorkingRuleSet:
    def test_the_rules_build_and_keep_their_identities(self):
        default_ids = [spec.rule_id for spec in rule_specs(DEFAULT_CONFIG)]
        preset_ids = [spec.rule_id for spec in rule_specs(LITERATURE_CONFIG)]
        assert default_ids == preset_ids

    def test_the_rules_carry_the_preset_thresholds_through(self):
        depth = next(s for s in rule_specs(LITERATURE_CONFIG) if s.rule_id == "squat_depth")
        assert depth.acceptable_max == LITERATURE_CONFIG.DEPTH_KNEE_ANGLE_PASS

    def test_every_rule_keeps_a_positive_tolerance(self):
        for spec in rule_specs(LITERATURE_CONFIG):
            assert spec.tolerance > 0, spec.rule_id
