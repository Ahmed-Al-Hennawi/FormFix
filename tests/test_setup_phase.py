"""
The setup before a set shouldn't be judged, and shouldn't calibrate the rules
either.

The rep state machine already only hands rules the frames inside a detected rep,
so the gap these cover is the standing baseline: it used to be measured over
every "standing" frame before the first rep, which is exactly when someone is
walking in, picking up a weight and shuffling their feet.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from analysis import pose_detector
from analysis.models import (
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_KNEE,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_KNEE,
    RuleStatus,
)
from exercises.squat.analyser import analyse_squat
from exercises.squat.config import DEFAULT_CONFIG
from tests.synthetic_squat import FPS, SyntheticSpec, make_pose_data, write_plain_video


def add_setup_movement(pose, seconds: float = 1.0):
    """
    Make the standing frames at the start look like a real setup: the heels come
    off the floor, the knees bend a little and the body drifts sideways, the way
    someone does while they pick up a weight and find their stance.
    """
    frames = min(int(seconds * FPS), pose.frame_count)
    for f in range(frames):
        phase = f / max(frames - 1, 1)
        lift = 0.02 * np.sin(phase * np.pi * 2.0)
        for heel in (LEFT_HEEL, RIGHT_HEEL):
            pose.xy[f, heel, 1] -= lift
        for knee in (LEFT_KNEE, RIGHT_KNEE):
            pose.xy[f, knee, 1] += 0.01 * phase
        for hip in (LEFT_HIP, RIGHT_HIP):
            pose.xy[f, hip, 1] += 0.015 * phase
        pose.xy[f, :, 0] += 0.01 * phase
    pose.xy_raw[:] = pose.xy
    return pose


def run(tmp_path: Path, spec: SyntheticSpec, setup: bool = False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pose = make_pose_data(spec)
    if setup:
        add_setup_movement(pose)
    video = tmp_path / "clip.mp4"
    write_plain_video(video, pose.frame_count)
    with mock.patch.object(pose_detector, "detect_poses", return_value=pose):
        return analyse_squat(video, output_dir=tmp_path)


def rule(result, rule_id: str):
    return next(r for r in result.rule_results if r.rule_id == rule_id)


class TestTheBaselineIgnoresTheSetup:
    def test_it_reports_how_many_setup_frames_it_dropped(self, tmp_path):
        result = run(tmp_path, SyntheticSpec(standing_seconds=2.0, noise=0.002), setup=True)
        baseline = result.debug["standing_baseline"]
        assert baseline["setup_frames_skipped"] > 0
        assert baseline["standing_frames"] >= DEFAULT_CONFIG.BASELINE_MIN_FRAMES

    def test_a_restless_setup_does_not_move_the_baseline(self, tmp_path):
        spec = SyntheticSpec(standing_seconds=2.0, noise=0.002)
        still = run(tmp_path / "still", spec)
        restless = run(tmp_path / "restless", spec, setup=True)

        for key in ("knee_angle", "torso_lean"):
            assert still.debug["standing_baseline"][key] == pytest.approx(
                restless.debug["standing_baseline"][key], abs=3.0
            ), key

    def test_a_heel_raised_during_setup_is_not_a_heel_finding(self, tmp_path):
        result = run(tmp_path, SyntheticSpec(standing_seconds=2.0, noise=0.002), setup=True)
        assert rule(result, "heel_lift").status is not RuleStatus.FAIL
        assert rule(result, "heel_lift").status is not RuleStatus.WARNING


class TestRulesOnlyReadRepetitions:
    def test_no_finding_points_at_a_frame_before_the_first_repetition(self, tmp_path):
        result = run(tmp_path, SyntheticSpec(standing_seconds=2.0, noise=0.002), setup=True)
        first_rep_start = min(rep.start_frame for rep in result.reps)
        for rule_result in result.rule_results:
            for outcome in rule_result.per_rep:
                if outcome.status in (RuleStatus.FAIL, RuleStatus.WARNING):
                    assert outcome.evidence_frame >= first_rep_start
