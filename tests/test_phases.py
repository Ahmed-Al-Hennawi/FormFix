"""State-machine tests on synthetic knee-angle sequences."""

from __future__ import annotations

import numpy as np

from exercises.squat.config import SquatConfig
from exercises.squat.phases import detect_reps

CONFIG = SquatConfig()
FPS = 30.0


def ts(sequence: np.ndarray) -> np.ndarray:
    return np.arange(len(sequence)) / FPS


def one_rep(depth: float = 95.0, half_frames: int = 25) -> np.ndarray:
    down = np.linspace(175.0, depth, half_frames)
    hold = np.full(6, depth)
    up = np.linspace(depth, 175.0, half_frames)
    return np.concatenate([down, hold, up])


def standing(frames: int = 20, angle: float = 175.0) -> np.ndarray:
    return np.full(frames, angle)


class TestCompleteReps:
    def test_single_rep_counts_once(self):
        seq = np.concatenate([standing(), one_rep(), standing()])
        result = detect_reps(seq, ts(seq), CONFIG)
        assert len(result.reps) == 1
        assert result.partial_movements == 0

    def test_brief_style_sequence(self):
        # the example sequence, slowed down - as given its 13 values last 0.4s
        base = [175, 170, 160, 145, 130, 110, 95, 100, 115, 135, 155, 170, 175]
        seq = np.concatenate([standing(15)] + [np.full(4, v) for v in base] + [standing(15)])
        result = detect_reps(seq, ts(seq), CONFIG)
        assert len(result.reps) == 1

    def test_two_reps(self):
        seq = np.concatenate([standing(), one_rep(), standing(12), one_rep(), standing()])
        result = detect_reps(seq, ts(seq), CONFIG)
        assert len(result.reps) == 2
        first, second = result.reps
        assert first.end_frame < second.start_frame
        assert first.start_frame < first.bottom_frame < first.end_frame

    def test_rep_metadata(self):
        seq = np.concatenate([standing(), one_rep(depth=92.0), standing()])
        result = detect_reps(seq, ts(seq), CONFIG)
        rep = result.reps[0]
        assert rep.min_knee_angle == np.min(seq)
        assert abs(rep.bottom_frame - int(np.argmin(seq))) <= 6


class TestRobustness:
    def test_threshold_jitter_does_not_double_count(self):
        rng = np.random.default_rng(7)
        seq = np.concatenate([standing(), one_rep(), standing()])
        noisy = seq + rng.normal(0, 2.0, len(seq))
        result = detect_reps(noisy, ts(noisy), CONFIG)
        assert len(result.reps) == 1

    def test_hovering_at_threshold_is_not_reps(self):
        # +-3 deg around the rep-start threshold
        rng = np.random.default_rng(3)
        seq = CONFIG.REP_START_KNEE_ANGLE + rng.normal(0, 3.0, 300)
        result = detect_reps(seq, ts(seq), CONFIG)
        assert len(result.reps) == 0

    def test_shallow_dip_is_partial_not_rep(self):
        dip = np.concatenate(
            [standing(), np.linspace(175, 140, 15), np.linspace(140, 175, 15), standing()]
        )
        result = detect_reps(dip, ts(dip), CONFIG)
        assert len(result.reps) == 0
        assert result.partial_movements == 1

    def test_extremely_fast_movement_rejected(self):
        # a 4-frame squat is 0.13s, so it's jitter
        blip = np.concatenate([standing(), [150, 95, 95, 150], standing()])
        result = detect_reps(np.asarray(blip, dtype=float), ts(np.asarray(blip)), CONFIG)
        assert len(result.reps) == 0

    def test_no_movement_no_reps(self):
        seq = standing(200)
        result = detect_reps(seq, ts(seq), CONFIG)
        assert len(result.reps) == 0
        assert result.partial_movements == 0


class TestPartialRecordings:
    def test_video_starting_mid_squat(self):
        seq = np.concatenate([np.linspace(95, 175, 25), standing(), one_rep(), standing()])
        result = detect_reps(seq, ts(seq), CONFIG)
        # the half-seen first movement doesn't count, the full one does
        assert len(result.reps) == 1

    def test_video_ending_mid_squat(self):
        seq = np.concatenate([standing(), one_rep(), standing(10), np.linspace(175, 95, 25)])
        result = detect_reps(seq, ts(seq), CONFIG)
        assert len(result.reps) == 1
        assert result.partial_movements == 1

    def test_tracking_loss_mid_rep_abandons(self):
        rep = one_rep()
        seq = np.concatenate([standing(), rep[:20], np.full(40, np.nan), standing()])
        result = detect_reps(seq, ts(seq), CONFIG)
        assert len(result.reps) == 0
        assert result.partial_movements == 1

    def test_short_nan_gap_survives(self):
        rep = one_rep()
        rep[28:31] = np.nan  # 3 missing frames near the bottom
        seq = np.concatenate([standing(), rep, standing()])
        result = detect_reps(seq, ts(seq), CONFIG)
        assert len(result.reps) == 1

    def test_bounce_at_bottom_is_one_rep(self):
        down = np.linspace(175, 95, 25)
        bounce = np.array([97, 99, 96, 94, 95, 97])
        up = np.linspace(97, 175, 25)
        seq = np.concatenate([standing(), down, bounce, up, standing()])
        result = detect_reps(seq, ts(seq), CONFIG)
        assert len(result.reps) == 1
