# Research provenance

What I took from the papers I reviewed, what I changed or extended, and what is
my own. [literature_basis.md](literature_basis.md) covers the same papers linked
to the code; this file is the provenance record.

Three things up front:

- **Nothing was copied.** No code, figures or sentences from the papers are in
  FormFix. What I took were measurements, criteria and reasoning, and each one
  is cited.
- **Some of my decisions turned out to match published work I hadn't read yet.**
  Those are labelled CONVERGENT - they are agreement, not a source.
- **My main new contribution is narrow:** using the published measurement error
  inside the decision rule (§4).

Paper numbers follow the files I was given: 1, 2 and 4-9 (there is no paper 3).
Jaiswal et al. (2023) was added later and has no number.

---

## 1. The papers

| # | Reference | What it is |
| --- | --- | --- |
| 1 | Kotte, H., Kravčík, M. and Duong-Trung, N. (2023) 'Real-time posture correction in gym exercises: a computer vision-based approach for performance analysis, error classification and feedback', *MILeS'23, ECTEL 2023*, CEUR Workshop Proceedings, vol. 3499. | YOLOv7-pose gym feedback; angle ranges set by an expert. |
| 2 | Mercadal-Baudart, C. et al. (2024) 'Exercise quantification from single camera view markerless 3D pose estimation', *Heliyon*, 10(6), e27596. doi:10.1016/j.heliyon.2024.e27596 | Physio metrics from single-camera 3D pose, tested against VICON. |
| 4 | Chae, H.J. et al. (2023) 'An artificial intelligence exercise coaching mobile app: development and randomized controlled trial to verify its effectiveness in posture correction', *Interactive Journal of Medical Research*, 12, e37604. doi:10.2196/37604 | CNN+LSTM squat classifier and a two-week randomised trial. |
| 5 | Simoes, W. et al. (2024) 'Accuracy assessment of 2D pose estimation with MediaPipe for physiotherapy exercises', *Procedia Computer Science*, 251, pp. 446-453. doi:10.1016/j.procs.2024.11.132 | MediaPipe + Naïve Bayes with a user-set tolerance. |
| 6 | Rao, P., Asha, C.S. and Raghavendra Rao, P. (2025) 'Real-time posture correction of squat exercise: a deep learning approach for performance analysis and error correction', *IEEE Access*, 13, pp. 39557-39571. doi:10.1109/ACCESS.2025.3545207 | Stereo squat classification; quotes squat angle ranges from Kotiuk et al. (2022). |
| 7 | Dill, S. et al. (2023) 'Accuracy evaluation of 3D pose estimation with MediaPipe Pose for physical exercises', *Current Directions in Biomedical Engineering*, 9(1), pp. 563-566. doi:10.1515/cdbme-2023-1141 | MediaPipe vs Xsens: joint-angle error and body-width instability by camera angle. |
| 8 | Dill, S. et al. (2024) 'Accuracy evaluation of 3D pose reconstruction algorithms through stereo camera information fusion for physical exercises with MediaPipe Pose', *Sensors*, 24(23), 7772. doi:10.3390/s24237772 | 810 squat reps vs Qualisys: per-limb knee error, a filter comparison, a definition of a correct squat. |
| 9 | Yadav, S.K. et al. (2022) 'YogNet: a two-stream network for realtime multiperson yoga action recognition and posture correction', *Knowledge-Based Systems*, 250, 109097. doi:10.1016/j.knosys.2022.109097 | Yoga recognition with an angle-threshold correction table. |
| - | Jaiswal, A., Chauhan, G. and Srivastava, N. (2023) 'Using learnable physics for real-time exercise form recommendations', *RecSys '23*. ACM. doi:10.1145/3604915.3608816 | Learns a physics model per exercise; argues against hand-set thresholds. |

Kotiuk et al. (2022) is cited through Rao et al. (2025) - I haven't read the
original, so it is cited that way.

---

## 2. What came from where

How to read the last column:

- **APPLIED** - used as published, with a citation.
- **ADAPTED** - used but changed for a stated reason (recorded in the code).
- **EXTENSION** - the paper was the starting point; FormFix does something it
  doesn't.
- **CONVERGENT** - I was already doing it before I read the paper.
- **ORIGINAL** - no paper contributed this.

| Part of FormFix | Ref. | What I took | Type and how it differs |
| --- | --- | --- | --- |
| Measurement uncertainty (`common/uncertainty.py`) | 7, 8 | Error figures: 10.7° near leg, 25.1° far leg (8); 9.14° good angle, 14.48° bad angle, 5-34% body-width instability (7). | **EXTENSION.** The papers report accuracy but don't use it in a decision. FormFix marks findings inside the error as indicative. |
| Acceptability bar (`ACCEPTABILITY_THRESHOLDS`) | 2 | Under 12° = good (better than a physio by eye), under 6° = very good. | **APPLIED** to FormFix's own numbers: trunk and near-knee pass, left/right difference and far knee don't. |
| Goniometry benchmark | 8 | Hancock et al. (2018), quoted in 8: 6° / 10° / 14° on a still patient. | **APPLIED** as context for the 10.7° band. |
| Far-leg exclusion | 8 | The far leg has 25.1° error vs 10.7°. | **CONVERGENT.** I already analysed the more visible side; this gives the number behind it. |
| Filter comparison (`filters.py`, `compare_filters.py`) | 8 | The lag problem and the three filters; their choice of a 2 Hz Butterworth. | **EXTENSION.** On my criterion (depth bias per rep) the ranking flips with rep shape, so the filter is a setting and its bias goes into the depth error. |
| Filter implementation | - | Nothing - standard signal processing. | **ORIGINAL.** Written in NumPy to avoid SciPy; tested against known maths. |
| Display smoothing of the result skeleton | 8 | The 2 Hz Butterworth settings. | **APPLIED**, for drawing only - it doesn't change any measurement. |
| Literature preset (`squat/literature_config.py`) | 5, 6, 8 | Kotiuk et al. squat angles via 6; the ~20° gap between correct and faulty reps (8); the 20° beginner tolerance (5). | **ADAPTED** and not the default: flexion converted to interior angle, and supervised lab squats aren't phone videos. For comparison only. |
| Squat rules | 8 | Their correct squat: heels down, straight spine, even loading; faults: leaning forward, shifting sideways. | **CONVERGENT.** My heel, lean and depth rules were written before I read it. No rule changed because of it. |
| Trunk angle vs your own standing posture | 2 | Trunk angle is 0° when standing. | **CONVERGENT** - I did it for camera tilt; now it has a citation. |
| Mid-hip / mid-shoulder points | 9 | Averaging keypoints that should line up in a side view. | **CONVERGENT.** |
| Rule spec (`common/spec.py`) | 9, 1 | Threshold + fixed correction (9); expert-set angle ranges (1). | **CONVERGENT + EXTENSION.** My rules also store phase, camera views, persistence and a source, so they can say what they *can't* assess. |
| Per-exercise landmarks in the result video | - | Jaiswal et al. also pick landmarks per exercise. | **CONVERGENT.** |
| Normalisation and smoothing | 4 | Standing reference frame, torso-length normalisation, per-keypoint smoothing. | **CONVERGENT.** No code taken. |
| Not claiming knee valgus or back rounding | 2 | Nothing - they can measure valgus because they lift to 3D. | **ORIGINAL position.** One 2D camera can't see it, so FormFix doesn't claim it. |
| No efficacy claim | 4 | A randomised controlled trial as the standard. | Used as a limitation - FormFix hasn't had a user study. |
| Per-user strictness | 5 | User-chosen tolerance. | Not implemented - named as a gap. |
| Hand-set rules vs learned models | - | Jaiswal et al.'s criticism of hand-crafted parameters. | Discussed as the main trade-off in the thesis. |

---

## 3. What I added to the code because of the review

All of these were additive - no existing verdict, score or threshold changed at
the time, and the existing tests still passed.

- **Measurement uncertainty layer** - `exercises/common/uncertainty.py`, hooked
  into all three analysers. Default only annotates; `UNCERTAINTY_STRICT` also
  downgrades.
- **Filters and a comparison script** - `analysis/filters.py`,
  `scripts/compare_filters.py`.
- **Literature preset** - `exercises/squat/literature_config.py`, run with
  `--preset literature`.
- **Settings** - `ANGLE_FILTER`, `UNCERTAINTY_STRICT`, and the matching
  evaluation options.
- **Tests and docs** - `test_filters.py`, `test_uncertainty.py`,
  `test_literature_preset.py`, and
  [measurement_uncertainty.md](measurement_uncertainty.md),
  [filter_selection.md](filter_selection.md),
  [literature_basis.md](literature_basis.md).

---

## 4. What the review showed

These come from applying the papers' numbers to my own thresholds, so none of
the papers contain them:

| FormFix measurement | Error | Source | Against the 12° / 6° bar |
| --- | --- | --- | --- |
| Trunk angle | ±6.5° | Derived from paper 8's 56.3 mm landmark error | good |
| Knee angle, near leg (depth) | ±10.7° (±11.6° with filter bias) | Paper 8 | good |
| Left/right knee difference | ±15.1° | Derived: √2 × 10.7° | outside the band |
| Knee angle, far leg | ±25.1° | Paper 8 | outside the band |
| Normalised length (heel lift) | ±0.15 | Derived from paper 7's 5-34% | not an angle |

- The squat left/right warning (12°) sat below its own error (±15.1°). I later
  removed that rule; the measurement is still exported.
- The heel threshold (0.06) sat below the ±0.15 error of what it is divided by.
  It is now 0.16 in the defaults as well as the literature preset, clear of the
  band.
- The depth bands are sized sensibly - anything bad enough to fail is always
  outside the error, so a depth fail is never borderline.
- The best filter depends on the rep shape: the 2 Hz Butterworth is nearly
  perfect on a smooth rep (+0.04°) and worst on a sharp one (+7.67°).
- The three policies disagree by 17 points on the same three 105° squats:
  default 90 (warning), strict 100 (pass), literature preset 83 (fail).

### My original contribution, kept narrow

- **Using published measurement error in the decision.** Several papers report
  accuracy and several use fixed thresholds, but none connects them.
- **Refusing to assess instead of guessing.** Each rule says which camera angles
  it works from and reports "not assessed" otherwise.

---

## 5. Where to find it in the project

| What | Where |
| --- | --- |
| Error bands, sources and derivations | `exercises/common/uncertainty.py` |
| The filters | `analysis/filters.py` |
| The filter comparison | `scripts/compare_filters.py`, [filter_selection.md](filter_selection.md) |
| The literature preset | `exercises/squat/literature_config.py` |
| Paper-by-paper notes | [literature_basis.md](literature_basis.md) |
| Threshold history | [threshold_tuning.md](threshold_tuning.md) |
| Limitations | [limitations.md](limitations.md) |
| Tests | `tests/test_filters.py`, `test_uncertainty.py`, `test_literature_preset.py` |
