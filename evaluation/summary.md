# FormFix technical evaluation

Recordings in the manifest: **14**  
Analysed successfully: **13**  
Stopped before analysis: **1**

No accuracy percentage is computed here. The counts below are what the
system did; whether a sample of this size supports a rate is a question
for the evaluation chapter, not for a script.

## Per recording

| Video | Exercise | Expected | Detected | Reps | Score | Reliability | Recording |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LP01_correct.mp4 | pulldown | correct | pulldown_torso | 4 | 56 | high | good |
| LP02_excessive_lean.mp4 | pulldown | pulldown_torso | not analysed (no_complete_repetition) | - | - | - | - |
| LP03_limited_rom.mp4 | pulldown | pulldown_rom | correct | 2 | 100 | high | good |
| LP04_front_view.mp4 | pulldown | correct | correct | 2 | 100 | medium | limited |
| SP01_correct.mp4 | press | correct | correct | 4 | 100 | medium | limited |
| SP02_asymmetric.mp4 | press | press_symmetry | press_rom;press_symmetry | 4 | 67 | medium | limited |
| SP04_incomplete_rom.mp4 | press | press_rom | press_rom | 3 | 67 | high | good |
| SP05_side_view.mp4 | press | correct | correct | 4 | 100 | medium | limited |
| SQ01a_correct_weighted.mp4 | squat | correct | correct | 3 | 100 | high | good |
| SQ01b_correct_bodyweight.mp4 | squat | correct | correct | 2 | 100 | high | good |
| SQ02_to_parallel.mp4 | squat | correct | correct | 1 | 100 | medium | good |
| SQ03_heel_lift.mp4 | squat | heel_lift | correct | 1 | 100 | medium | good |
| SQ04_setup_heel_lift.mp4 | squat | heel_lift | heel_lift | 2 | 90 | high | good |
| XX01_wrong_exercise.mp4 | press | correct | press_rom | 1 | 50 | medium | limited |

## Per rule

TP - the rule fired on a recording labelled with that fault.  
FP - it fired on a recording not labelled with it.  
FN - it stayed silent on a recording labelled with it.  
TN - it correctly stayed silent.  
NOT ASSESSED - the recording could not support the measurement, so the
system declined to judge it. That is not a miss and is counted apart.

| Rule | TP | FP | FN | TN | Not assessed |
| --- | --- | --- | --- | --- | --- |
| pulldown_rom | 0 | 0 | 1 | 2 | 0 |
| pulldown_torso | 0 | 1 | 0 | 2 | 0 |
| press_symmetry | 1 | 0 | 0 | 2 | 2 |
| press_alignment | 0 | 0 | 0 | 3 | 2 |
| press_rom | 1 | 2 | 0 | 2 | 0 |
| squat_depth | 0 | 0 | 0 | 5 | 0 |
| torso_lean | 0 | 0 | 0 | 5 | 0 |
| heel_lift | 1 | 0 | 1 | 3 | 0 |
| return_to_standing | 0 | 0 | 0 | 5 | 0 |
| descent_control | 0 | 0 | 0 | 5 | 0 |

## Recordings that were not analysed

- **LP02_excessive_lean.mp4** - no_complete_repetition: We couldn't detect a complete lat pulldown clearly in this video. 2 partial movement(s) were seen but did not qualify as full repetitions.

## Not-assessed measurements

- **SP05_side_view.mp4** (side): press_alignment, press_symmetry
- **XX01_wrong_exercise.mp4** (side): press_alignment, press_symmetry
