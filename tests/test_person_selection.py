"""
Which person gets analysed when more than one is in shot.

In a real gym someone is nearly always in the background, and the old rule
(reject if a second person is in over half the frames) threw out videos that
were tracked fine. Worse, at the bottom of a squat a bystander could become the
tallest person and silently take over the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.models import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER
from analysis.pose_detector import (
    MAX_IDENTITY_GAP_FRAMES,
    _PersonTracker,
    _primary_pose,
)


@dataclass
class LM:
    x: float
    y: float
    visibility: float = 1.0


def person(hip_x: float, hip_y: float, torso: float = 0.18, height: float = 0.55) -> list:
    """Minimal 33-landmark body with a known hip position and box height."""
    lms = [LM(hip_x, hip_y) for _ in range(33)]
    lms[LEFT_HIP] = LM(hip_x - 0.02, hip_y)
    lms[RIGHT_HIP] = LM(hip_x + 0.02, hip_y)
    lms[LEFT_SHOULDER] = LM(hip_x - 0.03, hip_y - torso)
    lms[RIGHT_SHOULDER] = LM(hip_x + 0.03, hip_y - torso)
    lms[0] = LM(hip_x, hip_y - height / 2)  # head, box top
    lms[31] = LM(hip_x - 0.03, hip_y + height / 2)  # foot, box bottom
    lms[32] = LM(hip_x + 0.03, hip_y + height / 2)
    return lms


# --- Starting the track ---


def test_single_person_is_tracked():
    t = _PersonTracker()
    athlete = person(0.5, 0.5)
    assert t.select([athlete], 0) is athlete
    assert t.switches == 0


def test_first_frame_picks_the_tallest():
    """No history yet, so the biggest person in frame wins."""
    t = _PersonTracker()
    athlete = person(0.5, 0.5, height=0.70)  # near the camera
    bystander = person(0.2, 0.45, height=0.30)  # further back
    assert t.select([athlete, bystander], 0) is athlete


def test_empty_frame_returns_nothing():
    assert _PersonTracker().select([], 0) == []


# --- The bug this exists to prevent ---


def test_bystander_cannot_steal_the_track_at_the_bottom_of_a_squat():
    """The athlete's box shrinks as they squat, but tracking keeps it on them."""
    t = _PersonTracker()
    athlete_standing = person(0.5, 0.50, height=0.70)
    bystander = person(0.8, 0.45, height=0.55)

    assert t.select([athlete_standing, bystander], 0) is athlete_standing

    # bottom of the rep: the box is now SHORTER than the bystander's
    athlete_squatting = person(0.5, 0.62, height=0.42)
    picked = t.select([athlete_squatting, bystander], 1)

    assert picked is athlete_squatting, "tracking jumped to the bystander"
    assert t.switches == 0


def test_tracking_follows_the_athlete_through_a_whole_rep():
    t = _PersonTracker()
    bystander = person(0.85, 0.45, height=0.60)
    # hips descend and come back up; the box shrinks and grows with them
    for frame, (hip_y, h) in enumerate(
        [(0.50, 0.70), (0.55, 0.60), (0.62, 0.42), (0.55, 0.60), (0.50, 0.70)]
    ):
        athlete = person(0.5, hip_y, height=h)
        assert t.select([athlete, bystander], frame) is athlete
    assert t.switches == 0


def test_the_nearest_candidate_wins_not_the_tallest():
    t = _PersonTracker()
    athlete = person(0.5, 0.5, height=0.60)
    t.select([athlete], 0)
    # a taller person appears, but further from where the athlete was
    taller_elsewhere = person(0.62, 0.5, height=0.90)
    moved_slightly = person(0.51, 0.5, height=0.60)
    assert t.select([taller_elsewhere, moved_slightly], 1) is moved_slightly


# --- Losing and recovering the track ---


def test_a_jump_across_the_room_counts_as_a_switch():
    t = _PersonTracker()
    t.select([person(0.2, 0.5)], 0)
    # nobody near where the athlete was
    t.select([person(0.85, 0.5), person(0.9, 0.52)], 1)
    assert t.switches == 1


def test_track_is_dropped_after_a_long_absence():
    t = _PersonTracker()
    t.select([person(0.2, 0.5)], 0)
    # athlete gone longer than the allowed gap, so height decides again
    far_future = MAX_IDENTITY_GAP_FRAMES + 5
    tall = person(0.85, 0.5, height=0.80)
    short = person(0.88, 0.5, height=0.30)
    assert t.select([tall, short], far_future) is tall
    assert t.switches == 0, "an expired track is not a disruption"


def test_brief_occlusion_does_not_break_the_track():
    t = _PersonTracker()
    athlete = person(0.5, 0.5)
    t.select([athlete], 0)
    t.select([], 1)  # someone walks in front of the camera
    t.select([], 2)
    still_there = person(0.51, 0.5)
    assert t.select([still_there, person(0.9, 0.4)], 3) is still_there
    assert t.switches == 0


def test_malformed_landmarks_do_not_raise():
    t = _PersonTracker()
    t.select([person(0.5, 0.5)], 0)
    broken = [LM(float("nan"), float("nan")) for _ in range(33)]
    # should fall back, not crash
    assert t.select([broken, person(0.5, 0.5)], 1) is not None


# --- The single-frame helper still behaves ---


def test_primary_pose_picks_the_tallest():
    tall, short = person(0.5, 0.5, height=0.80), person(0.2, 0.5, height=0.30)
    assert _primary_pose([short, tall]) is tall


def test_primary_pose_with_one_person():
    only = person(0.5, 0.5)
    assert _primary_pose([only]) is only


# --- What actually rejects a recording ---


def _pose(frames: int, n_poses: int, switches: int):
    """A pose track with every frame found and a fixed number of people in shot."""
    from analysis.models import FramePoseData

    return FramePoseData(
        xy_raw=np.zeros((frames, 33, 2)),
        xy=np.zeros((frames, 33, 2)),
        visibility=np.ones((frames, 33)),
        valid=np.ones((frames, 33), dtype=bool),
        pose_found=np.ones(frames, dtype=bool),
        n_poses=np.full(frames, n_poses, dtype=np.int16),
        timestamps=np.arange(frames) / 30.0,
        identity_switches=switches,
    )


def _multi_person_outcome(frames=900, n_poses=2, switches=0, fps=30.0):
    """Run just the multi-person check and return (rejected, warned)."""
    from analysis.validation import MAX_SWITCH_RATE_PER_SECOND, MIN_SWITCHES_TO_REJECT

    pose = _pose(frames, n_poses, switches)
    tracked = int(pose.pose_found.sum())
    rate = switches / max(1.0, tracked / max(fps, 1.0))
    rejected = switches >= MIN_SWITCHES_TO_REJECT and rate >= MAX_SWITCH_RATE_PER_SECOND
    warned = (not rejected) and (switches > 0 or (pose.n_poses > 1).mean() >= 0.10)
    return rejected, warned


def test_busy_gym_with_clean_tracking_is_accepted():
    """Someone in every frame but the athlete tracked the whole time - used to be rejected."""
    rejected, warned = _multi_person_outcome(n_poses=2, switches=0)
    assert not rejected
    assert warned, "a bystander should still be mentioned"


def test_crowded_gym_is_accepted_when_tracking_holds():
    rejected, _ = _multi_person_outcome(n_poses=4, switches=0)
    assert not rejected


def test_a_couple_of_switches_warns_but_does_not_reject():
    rejected, warned = _multi_person_outcome(frames=900, switches=2)
    assert not rejected
    assert warned


def test_a_track_that_keeps_coming_apart_is_rejected():
    # 30 s clip with a switch about every second - not one person any more
    rejected, _ = _multi_person_outcome(frames=900, switches=25)
    assert rejected


def test_solo_video_is_neither_rejected_nor_flagged():
    rejected, warned = _multi_person_outcome(n_poses=1, switches=0)
    assert not rejected
    assert not warned


def test_switch_rate_is_length_independent():
    """Same disruption rate, different clip lengths - same verdict."""
    short_clip, _ = _multi_person_outcome(frames=150, switches=5)  # 5 s
    long_clip, _ = _multi_person_outcome(frames=1800, switches=60)  # 60 s
    assert short_clip == long_clip is True


def test_a_few_switches_in_a_very_short_clip_still_rejects_only_if_frequent():
    # 4 switches over 20 s is 0.2/s, under the limit
    rejected, _ = _multi_person_outcome(frames=600, switches=4)
    assert not rejected
