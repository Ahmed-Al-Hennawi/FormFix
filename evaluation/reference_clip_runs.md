# Real-video runs

These are real end-to-end runs of FormFix (real MediaPipe detection, not
synthetic data) on the three reference clips in the project.

**This is verification, not evaluation.** The reference clips are 3D-animated
instructional videos that cut between shots, close-ups and muscle overlays -
not one person doing a set. So they can't give an accuracy figure. What they do
show is that the pipeline works on real, messy footage and refuses to make up
findings when it can't see the movement.

To reproduce:

```bash
python scripts/calibrate.py static/assets/videos/lat-pulldown-reference.mp4 pulldown
```

Recorded 29 August 2026 with MediaPipe 0.10.14, NumPy 1.26.4, OpenCV 4.10,
Python 3.10 (CPU).

## 1. Squat - correctly refused

| Measure | Value |
| --- | --- |
| Pose coverage | 84.5% of frames |
| Usable frames (best side) | 83.0% |
| Camera | diagonal, side-view confidence 0.72 |
| Reps | **0 complete, 7 partial** |
| Result | `no_complete_squat` - stopped |

The figure was tracked well, but the clip keeps cutting between angles, so
seven squats started and none was seen all the way through. A montage isn't a
set, and FormFix said so instead of building a verdict from fragments.

## 2. Lat pulldown - analysed, with limits stated

| Measure | Value |
| --- | --- |
| Pose coverage | 62.4% of frames |
| Usable frames | 52.4% |
| Camera | side, confidence 1.00 |
| View stability | 0.62 (the angle changes) |
| Second person | 24.6% of frames |
| Reps | 1 complete, 1 partial (clip ends mid-rep) |
| Recording quality | limited |

| Rep | Top elbow | Bottom elbow | ROM | Torso movement | Wrist travel | Valid frames |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 177° | 42° | 135° | 18.9° | 0.68 | 95% |

| Rule | Result | Reliability | Evidence |
| --- | --- | --- | --- |
| `pulldown_rom` | **PASS** | medium | 177° top / 42° bottom vs ≥150° and ≤100° |
| `pulldown_torso` | **WARNING** | medium | 18.9° vs a 15° limit, held for **21 of 49** frames of the pull (43%) |

![Four frames of the annotated lat pulldown output](figures/pulldown_annotated_frames.png)

*Frames from this run: top position, the pull, the bottom position and the
return. The amber rings on the shoulder and hip mark the landmarks the torso
warning came from. This is the overlay as it was on 29 August. It has been
simplified twice since: first the phase marker and angle labels went, then in
September the rings and the colour changes went too, leaving one style - thin
cyan links, small white joints, the whole skeleton, and the camera-far side
drawn fainter.*

The torso warning can be traced step by step: shoulder and hip landmarks →
trunk angle → change from this person's own top position → 18.9° peak → held
for 43% of the pull → above 15° → warning, at medium reliability because the
recording was already marked as limited. The three warnings it raised (second
person, patchy tracking, changing angle) are all true of the clip.

## 3. Shoulder press - correctly refused

| Measure | Value |
| --- | --- |
| Pose coverage | 55.0% of frames |
| Usable frames (best side) | **33.8%** |
| Camera | side, frontality 0.19 |
| View stability | 0.47 (the angle changes) |
| Result | `important_landmarks_missing` - stopped |

About half the clip is close-ups with no arms in shot, so the arms were only
measurable in a third of the frames - not enough to judge anyone. It also
correctly noted that the press needs a front view.

## Bugs these runs found

Real footage hit code paths the synthetic tests didn't. All four were fixed
and now have regression tests:

1. **Skeleton missing on some frames.** The overlay used MediaPipe's raw
   "person found" flag, but the analysis also uses gap-filled frames, so on 21
   frames the video showed numbers with no skeleton
   (`TestOverlayShowsWhatTheAnalysisUsed`).
2. **Wrong phase caption.** An abandoned partial rep left its labels inside the
   next real rep, so the video said "RETURNING" at the start of a rep
   (`TestPhaseCaptionMatchesTheRepetition`).
3. **Pulldown ROM needed both arms.** From the side the far arm is hidden, so
   the check said "not assessed" on exactly the camera angle I recommend. It now
   needs only the analysed arm (before: `NOT_EVALUABLE`, after: `PASS`). The
   press alignment check had the same problem and was fixed the same way.
4. **Empty video output.** If the output folder didn't exist,
   `cv2.VideoWriter` silently dropped every frame. The folder is now created,
   and a writer that can't open raises an error (`TestWriterGuards`).

## Where this led

None of this validates a *threshold*. That needed the labelled videos in
[README.md](README.md), which were recorded in September 2026 and run through
`scripts/evaluate_videos.py` - the results are in [docs/evaluation_results.md](../docs/evaluation_results.md).
They found two thresholds sitting in the wrong place, a heel lift missed by a
persistence setting, and two recordings that got through a guard rail.
