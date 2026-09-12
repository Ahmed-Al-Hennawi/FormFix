# Literature basis

What I took from each paper I reviewed, what I did with it, and where FormFix
does something the paper doesn't. `FormFix_Research_Provenance.docx` has the
version written for the report; this one is linked to the code.

## The papers

| # | Paper | What it is |
| --- | --- | --- |
| 1 | Kotte, Kravčík & Duong-Trung (2023), *MILeS'23 / ECTEL 2023* | Gym posture feedback with YOLOv7-pose; joint-angle ranges set by an expert |
| 2 | Mercadal-Baudart et al. (2024), *Heliyon* 10:e27596 | Single-camera 3D pose for exercise, tested against VICON |
| 4 | Chae et al. (2023), *Interact J Med Res* 12:e37604 | Deep-learning squat app, tested in a two-week randomised controlled trial |
| 5 | Simoes et al. (2024), *Procedia Comput Sci* 251:446-453 | MediaPipe + Naïve Bayes for physiotherapy, with a user-set tolerance |
| 6 | Rao, Asha & Raghavendra Rao (2025), *IEEE Access* 13:39557-39571 | Stereo squat classification and correction; quotes squat angle ranges |
| 7 | Dill et al. (2023), *Curr Dir Biomed Eng* 9(1):563-566 | Accuracy of MediaPipe Pose against motion capture |
| 8 | Dill et al. (2024), *Sensors* 24(23):7772 | Stereo MediaPipe on 810 squat reps against motion capture |
| 9 | Yadav et al. (2022), *Knowledge-Based Systems* 250:109097 | YogNet: yoga recognition with an angle-threshold correction table |
| - | Jaiswal, Chauhan & Srivastava (2023), *RecSys '23* | Learns a physics model per exercise and compares your movement to it |

Numbers 1-9 follow the files I was given (there's no paper 3). Jaiswal et al.
was added later, so it has no number.

## What I took and what I did with it

### An acceptability bar - Mercadal-Baudart et al. (2024)

They say a metric is "good" if its error is under **12°** (better than a
physio judging by eye) and "very good" under **6°**.

- **Taken:** the bar itself (`ACCEPTABILITY_THRESHOLDS` in
  `exercises/common/uncertainty.py`). Every angle error in FormFix is labelled
  against it: trunk and near-knee angles are "good", the left/right difference
  and far-knee angle are "worse than by eye".
- **Taken:** measuring trunk angle as 0° when standing - which I was already
  doing, so now it has a source.
- **Not taken:** their ~1° accuracy figures. They come from a different,
  trained model, so quoting them next to MediaPipe would be borrowing a number
  from another system.
- **Contrast:** they measure knee valgus because they have a 3D model trained
  on marker data. FormFix has one 2D camera, so it doesn't.

### Measurement error - Dill et al. (2023, 2024)

Paper 7 gives a squat knee-angle error of 9.14° at a good angle and 14.48° at
a bad one, plus 5-34% error on body widths. Paper 8 gives 10.7° for the near
leg and 25.1° for the hidden far leg, and quotes Hancock et al.'s goniometry
figures (6° / 10° / 14°).

- **Taken:** the numbers, as the error bands in `uncertainty.py`.
- **Extended:** neither paper uses its error in a decision. FormFix marks any
  finding smaller than the error as "indicative"
  ([measurement_uncertainty.md](measurement_uncertainty.md)).
- **Found because of it:** my left/right threshold (12°) and heel threshold
  (0.06) were below their own noise. I removed the squat left/right rule, and
  later raised the heel threshold to 0.16, clear of the band.

### Filtering - Dill et al. (2024)

They found moving averages lag at the top and bottom of a squat and chose a
2 Hz Butterworth.

- **Taken:** the problem and the three filters to compare.
- **Extended:** I tested on my own criterion (depth bias per rep) and the
  ranking flips with the shape of the rep, so the filter became a setting and
  its bias goes into the depth error ([filter_selection.md](filter_selection.md)).
  The skeleton on the result video has its own display smoothing, using their
  filter at a higher cut-off (3 Hz) after a running median, since it only has
  to remove small jitter and a low cut-off flattened fast movement.

### Squat angle ranges - Rao et al. (2025), quoting Kotiuk et al. (2022)

Knee flexion 113 ± 7°, hip flexion 128 ± 9°, ankle 23 ± 6°.

- **Taken:** into `exercises/squat/literature_config.py`.
- **Not made the default:** they report flexion but I measure the interior
  angle (113° flexion = 67° interior), and they measured supervised parallel
  squats. 67° would fail almost every real video, so the preset uses the most
  permissive reading and is only for comparison.

### Correct-squat criteria - Dill et al. (2024)

Their correct squat: shoulder-width stance, heels down, straight spine, both
legs loaded evenly. Faults: leaning too far forward, shifting sideways.

- **Taken:** as confirmation. I'd already come up with heel, lean and depth
  checks on my own, and they match a lab protocol. I also measured even
  loading but it couldn't support a verdict from one camera.
- **Taken as a number:** correct reps were about 20° away from faulty ones,
  which sets the literature preset's warning band.

### Per-user tolerance - Simoes et al. (2024)

They let users pick their tolerance: 20° for beginners, under 10° for
experienced users.

- **Taken:** the reason my bands are wide, and the 20° as the preset's
  `FULL_EXTENSION_TOLERANCE`.
- **Not taken:** a strictness setting for users. That's a real gap
  ([limitations.md](limitations.md)).

### Threshold + instruction tables - Yadav et al. (2022), Kotte et al. (2023)

YogNet pairs each angle threshold with a fixed correction. Kotte et al. let an
expert set the angle ranges.

- **Taken:** confirmation that FormFix's `RuleSpec` (threshold, phase, camera
  views, feedback template together) is a normal design for this. Mine carries
  more per rule because it also has to say what it *can't* assess.
- **Taken as a technique:** Yadav et al. average keypoints that should line up
  in a side view. I use mid-hip and mid-shoulder for the trunk for the same
  reason.

### Learned exercise models - Jaiswal et al. (2023)

They argue that systems with hand-set thresholds need "considerable
hand-crafting" and don't generalise across exercises and people. That's a fair
criticism of FormFix. What a learned model can't do is say which landmark
caused a finding, say a measurement wasn't available, or use a published error
in its decision - that's the trade-off I discuss in the thesis. They also pick
landmarks per exercise, like FormFix's overlay does.

### Evaluation design - Chae et al. (2023)

A two-week randomised controlled trial with 20 people: the app group improved
significantly (p = .001), the video-only group didn't (p = .13).

- **Taken:** as the standard for "does it actually help anyone?".
- **Not done:** FormFix hasn't had a user study, so I make no claim that it
  improves technique. Their preprocessing (standing reference, per-keypoint
  smoothing) is similar to what I'd already done.

## What none of them do

1. **Use their own measurement error in the decision.** Several report accuracy
   and several use fixed thresholds, but none connects the two.
2. **Refuse to assess instead of guessing.** Every paper gives a verdict for
   whatever it's given. FormFix says which camera views each rule works for
   and reports "not assessed" otherwise.
