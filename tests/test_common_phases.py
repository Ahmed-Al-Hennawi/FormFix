"""
Tests for the shared rep state machine. The signals are high at rest and drop
into the rep, so they work for all three exercises.
"""

from __future__ import annotations

import numpy as np
import pytest

from exercises.common.phases import (
    MovementPhase,
    RepDetectionConfig,
    detect_repetitions,
)

FPS = 30.0

CONFIG = RepDetectionConfig(
    rest_level=160.0,
    start_level=150.0,
    extreme_level=120.0,
    end_level=155.0,
    min_phase_frames=3,
    reversal_delta=5.0,
    min_range=30.0,
    extreme_window_frames=3,
    min_duration=0.8,
    max_duration=12.0,
    max_tracking_loss_frames=10,
)


def timestamps(n: int) -> np.ndarray:
    return np.arange(n) / FPS


def cycle(rest: float, extreme: float, hold: int = 12, travel: int = 21, pause: int = 5):
    """One rest -> travel -> turning point -> return."""
    return (
        list(np.full(hold, rest))
        + list(np.linspace(rest, extreme, travel))
        + list(np.full(pause, extreme))
        + list(np.linspace(extreme, rest, travel))
    )


def signal(*chunks) -> np.ndarray:
    values: list[float] = []
    for chunk in chunks:
        values.extend(chunk)
    return np.asarray(values, dtype=np.float64)


def run(values: np.ndarray, config: RepDetectionConfig = CONFIG):
    return detect_repetitions(values, timestamps(len(values)), config)


class TestCompleteRepetitions:
    def test_a_single_clean_repetition_is_counted_once(self):
        result = run(signal(cycle(175.0, 90.0), np.full(12, 175.0)))
        assert len(result.reps) == 1
        assert result.partial_movements == 0

    def test_three_repetitions_are_counted_separately(self):
        result = run(signal(*[cycle(175.0, 90.0) for _ in range(3)], np.full(12, 175.0)))
        assert len(result.reps) == 3

    def test_the_turning_point_lands_inside_the_extreme_window(self):
        result = run(signal(cycle(175.0, 90.0), np.full(12, 175.0)))
        rep = result.reps[0]
        assert rep.start_frame < rep.extreme_frame < rep.end_frame
        assert result.phases[rep.extreme_frame] is MovementPhase.EXTREME

    def test_every_phase_is_reached(self):
        result = run(signal(cycle(175.0, 90.0), np.full(12, 175.0)))
        seen = {phase for phase in result.phases}
        assert MovementPhase.REST in seen
        assert MovementPhase.TOWARDS in seen
        assert MovementPhase.EXTREME in seen
        assert MovementPhase.RETURN in seen


class TestFalsePositiveProtection:
    """Things that must not be counted as repetitions."""

    def test_noise_around_the_start_threshold_is_not_a_repetition(self):
        # jitters around start_level the whole clip - simple threshold counting
        # would call this a dozen reps
        rng = np.random.default_rng(0)
        values = 150.0 + rng.normal(0, 3.0, 240)
        result = run(values)
        assert result.reps == []

    def test_a_shallow_dip_is_a_partial_not_a_repetition(self):
        # passes the start level but never reaches the extreme
        result = run(signal(cycle(175.0, 135.0), np.full(12, 175.0)))
        assert result.reps == []
        assert result.partial_movements == 1
        assert "too shallow" in result.partial_reasons[0]

    def test_a_one_frame_spike_does_not_start_a_repetition(self):
        values = signal(np.full(30, 175.0), [80.0], np.full(30, 175.0))  # a single impossible frame
        result = run(values)
        assert result.reps == []

    def test_a_movement_too_fast_to_be_controlled_is_a_partial(self):
        # a whole rep in well under min_duration
        result = run(
            signal(
                np.full(10, 175.0), cycle(175.0, 90.0, hold=0, travel=3, pause=1), np.full(12, 175.0)
            )
        )
        assert result.reps == []
        assert result.partial_movements == 1
        assert "too fast" in result.partial_reasons[0]

    def test_a_movement_with_too_little_range_is_a_partial(self):
        tight = RepDetectionConfig(**{**CONFIG.__dict__, "min_range": 80.0})
        result = run(signal(cycle(175.0, 118.0), np.full(12, 175.0)), tight)
        assert result.reps == []
        assert result.partial_movements == 1
        assert "range of motion" in result.partial_reasons[0]


class TestIncompleteRecordings:
    def test_a_clip_starting_mid_movement_does_not_invent_a_repetition(self):
        # starts already at the extreme, so the descent was never seen
        values = signal(np.full(8, 90.0), np.linspace(90.0, 175.0, 21), np.full(12, 175.0))
        result = run(values)
        assert result.reps == []
        assert result.partial_movements == 0  # nothing was started, so nothing is partial

    def test_a_clip_ending_mid_movement_reports_a_partial(self):
        values = signal(np.full(12, 175.0), np.linspace(175.0, 90.0, 21), np.full(6, 90.0))
        result = run(values)
        assert result.reps == []
        assert result.partial_movements == 1
        assert "ended before" in result.partial_reasons[0]

    def test_a_complete_repetition_before_a_truncated_one_still_counts(self):
        values = signal(
            cycle(175.0, 90.0),
            np.full(6, 175.0),
            np.linspace(175.0, 90.0, 21),  # second rep never finishes
        )
        result = run(values)
        assert len(result.reps) == 1
        assert result.partial_movements == 1


class TestMissingData:
    def test_a_short_tracking_gap_does_not_break_a_repetition(self):
        # five missing frames mid-rep - brief occlusion shouldn't cost a rep
        values = signal(cycle(175.0, 90.0), np.full(12, 175.0))
        values[22:27] = np.nan
        result = run(values)
        assert len(result.reps) == 1

    def test_a_long_tracking_gap_abandons_the_repetition(self):
        # 17 missing frames is past max_tracking_loss_frames, so it's partial
        values = signal(cycle(175.0, 90.0), np.full(12, 175.0))
        values[25:42] = np.nan
        result = run(values)
        assert result.reps == []
        assert result.partial_movements == 1
        assert "tracking was lost" in result.partial_reasons[0]

    def test_an_empty_signal_is_handled(self):
        result = run(np.array([], dtype=np.float64))
        assert result.reps == []
        assert result.phases == []

    def test_an_all_missing_signal_produces_nothing(self):
        result = run(np.full(90, np.nan))
        assert result.reps == []
        assert result.partial_movements == 0


class TestHysteresisAndReversal:
    def test_a_bounce_at_the_turning_point_does_not_split_one_rep_in_two(self):
        # rises a bit then goes deeper - one rep with a wobble, not two
        values = signal(
            np.full(12, 175.0),
            np.linspace(175.0, 95.0, 18),
            [100.0, 104.0, 99.0, 92.0, 90.0],  # bounce, then deeper
            np.full(4, 90.0),
            np.linspace(90.0, 175.0, 21),
            np.full(12, 175.0),
        )
        result = run(values)
        assert len(result.reps) == 1

    def test_a_wobble_below_the_end_level_does_not_finish_the_rep_early(self):
        values = signal(
            np.full(12, 175.0),
            np.linspace(175.0, 90.0, 18),
            np.full(4, 90.0),
            np.linspace(90.0, 153.0, 12),  # rises to just under end_level
            np.full(4, 153.0),
            np.linspace(153.0, 175.0, 8),
            np.full(12, 175.0),
        )
        result = run(values)
        assert len(result.reps) == 1
        # ends where it actually passed end_level, not where it first got close
        assert result.reps[0].end_frame > 12 + 18 + 4 + 12

    def test_the_reported_excursion_is_measured_not_assumed(self):
        result = run(signal(cycle(175.0, 90.0), np.full(12, 175.0)))
        rep = result.reps[0]
        assert rep.extreme_value == pytest.approx(90.0, abs=1.0)
        assert rep.excursion > CONFIG.min_range
