"""
Tests for the render-window trimmer.

The behaviour that matters most here isn't the happy path - it's that every
odd input still produces a window that renders something. A trim that returns
an empty or backwards range would hand the user a zero-length video and read
as a broken analysis, so most of these tests are about failing open.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from analysis.trimming import RenderWindow, compute_render_window, full_video

FPS = 30.0


@dataclass
class FakeRep:
    """Stands in for RawRep / SquatRep / PressRep / PulldownRep."""

    start_frame: int
    end_frame: int


def window(reps, frame_count=900, fps=FPS, **kw):
    return compute_render_window(reps, frame_count, fps, **kw)


# --- The normal case ---


def test_trims_to_the_set_with_one_second_of_padding():
    # 30 s clip; the set runs from 10 s to 20 s.
    reps = [FakeRep(300, 450), FakeRep(450, 600)]
    result = window(reps)

    assert result.trimmed
    assert result.start_frame == 300 - 30  # one second before the first rep
    assert result.end_frame == 600 + 30  # one second after the last
    assert result.frame_count == 361


def test_padding_is_configurable():
    reps = [FakeRep(300, 600)]
    result = window(reps, padding_seconds=2.0)
    assert result.start_frame == 300 - 60
    assert result.end_frame == 600 + 60


def test_reports_seconds_saved():
    reps = [FakeRep(300, 600)]
    result = window(reps)
    # 900 frames in, 361 rendered -> 539 frames skipped at 30 fps
    assert result.seconds_saved(900, FPS) == pytest.approx(539 / 30.0, abs=0.01)


def test_untrimmed_window_saves_nothing():
    assert full_video(900).seconds_saved(900, FPS) == 0.0


# --- Failing open - every one of these must render the whole video ---


def test_no_reps_renders_everything():
    result = window([])
    assert not result.trimmed
    assert (result.start_frame, result.end_frame) == (0, 899)


def test_zero_frame_video_renders_everything():
    result = window([FakeRep(0, 10)], frame_count=0)
    assert not result.trimmed


def test_missing_fps_renders_everything():
    result = window([FakeRep(300, 600)], fps=0.0)
    assert not result.trimmed


def test_unsegmented_reps_render_everything():
    # -1 is the "never segmented" sentinel on the upper-body reps.
    result = window([FakeRep(-1, -1)])
    assert not result.trimmed
    assert "usable frame numbers" in result.reason


def test_rep_frames_beyond_the_video_render_everything():
    result = window([FakeRep(5000, 6000)], frame_count=900)
    assert not result.trimmed


def test_set_filling_the_clip_is_left_alone():
    # Nothing to cut: the reps already span the whole recording.
    result = window([FakeRep(0, 899)])
    assert not result.trimmed
    assert "not enough setup footage" in result.reason


def test_small_saving_is_not_worth_trimming():
    # Only ~1 s of setup either side, under the 2 s threshold.
    reps = [FakeRep(45, 855)]
    result = window(reps)
    assert not result.trimmed


def test_very_short_window_renders_everything():
    # A one-frame "rep" with no padding would give a sub-second clip.
    result = window([FakeRep(400, 401)], padding_seconds=0.0)
    assert not result.trimmed
    assert "too short" in result.reason


# --- Boundaries and odd ordering ---


def test_padding_is_clamped_to_the_video():
    # A set starting at frame 5 can't be padded to -25.
    result = window([FakeRep(5, 500)])
    assert result.start_frame == 0


def test_end_padding_is_clamped_to_the_last_frame():
    result = window([FakeRep(100, 890)], frame_count=900)
    assert result.end_frame == 899


def test_reps_out_of_order_still_span_correctly():
    reps = [FakeRep(600, 750), FakeRep(200, 350)]
    result = window(reps)
    assert result.start_frame == 200 - 30
    assert result.end_frame == 750 + 30


def test_reversed_rep_boundaries_are_normalised():
    result = window([FakeRep(600, 300)])
    assert result.start_frame == 300 - 30
    assert result.end_frame == 600 + 30


def test_single_rep_trims():
    result = window([FakeRep(400, 500)])
    assert result.trimmed
    assert result.start_frame == 370
    assert result.end_frame == 530


def test_reps_with_unreadable_frames_are_skipped_not_fatal():
    class Broken:
        @property
        def start_frame(self):
            raise RuntimeError("boom")

        @property
        def end_frame(self):
            return 500

    result = window([Broken(), FakeRep(300, 600)])
    # The broken rep is ignored; the usable one still defines the window.
    assert result.trimmed
    assert result.start_frame == 270


def test_non_integer_frames_are_ignored():
    result = window([FakeRep("nope", None), FakeRep(300, 600)])
    assert result.trimmed
    assert result.start_frame == 270


# --- The window object itself ---


def test_contains():
    w = RenderWindow(100, 200, trimmed=True)
    assert w.contains(100) and w.contains(200) and w.contains(150)
    assert not w.contains(99)
    assert not w.contains(201)


def test_as_dict_carries_what_the_export_needs():
    w = compute_render_window([FakeRep(300, 600)], 900, FPS)
    data = w.as_dict()
    assert data["trimmed"] is True
    assert data["start_frame"] == 270
    assert data["end_frame"] == 630
    assert data["frame_count"] == 361
    assert data["reason"]


def test_full_video_window_is_never_empty():
    assert full_video(1).frame_count == 1
    assert full_video(0).frame_count == 1


@pytest.mark.parametrize("frames", [1, 2, 30, 900, 3600])
def test_window_is_always_renderable(frames):
    """Whatever comes in, the window must address at least one real frame."""
    for reps in ([], [FakeRep(0, 0)], [FakeRep(-1, -1)], [FakeRep(0, frames * 3)]):
        w = compute_render_window(reps, frames, FPS)
        assert 0 <= w.start_frame <= w.end_frame <= max(0, frames - 1)
        assert w.frame_count >= 1


# --- The renderer's own clamp ---


def test_resolve_range_defaults_to_the_whole_video():
    from analysis.annotation import _resolve_range

    assert _resolve_range(None, 900) == (0, 899)


@pytest.mark.parametrize(
    "given",
    [
        (-50, 5000),  # both ends out of bounds
        (700, 300),  # reversed
        (5000, 6000),  # entirely past the end
        ("a", "b"),  # not numbers
        (None, None),
        (),  # too short to unpack
    ],
)
def test_resolve_range_never_returns_an_empty_window(given):
    from analysis.annotation import _resolve_range

    first, last = _resolve_range(given, 900)
    assert 0 <= first <= last <= 899


def test_resolve_range_keeps_a_valid_window():
    from analysis.annotation import _resolve_range

    assert _resolve_range((270, 630), 900) == (270, 630)
