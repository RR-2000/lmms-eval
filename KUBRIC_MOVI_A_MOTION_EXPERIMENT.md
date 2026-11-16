# MOVi-A Motion Diagnostics

## What this experiment asks

This experiment tests whether a vision-language model can reason about motion in short synthetic videos. Each video contains rigid objects moving and colliding in a 3D scene. The model receives an ordered sequence of rendered frames and a five-option multiple-choice question, then must return only `A`, `B`, `C`, `D`, or `E`.

The central question is not merely **“Did the model answer correctly?”** It is:

> When a model fails a motion question, did it fail to see and track the object, identify the relevant time interval, interpret the coordinate system, perform the calculation, use the supplied data representation, or remain robust to misleading evidence?

The experiment answers this by rerunning the same source question under controlled diagnostic scaffolds. Because the underlying video, question, options, and correct answer remain paired, the change from baseline accuracy to assisted accuracy helps localize the bottleneck.

## The visual input

The data comes from MOVi-A at `256 × 256` resolution. A sequence contains 24 frames sampled at 12 FPS, representing two seconds of simulated motion. The generation pipeline also saves object metadata, segmentation masks, world coordinates, camera coordinates, velocities, visibility, and collision information.

The following example contains five objects. Read left-to-right, then continue on the second row.

![Six frames from MOVi-A sequence 1](docs/assets/kubric_movi_a_motion/sequence_1_timeline.png)

Frames shown: 0, 4, 8, 12, 16, and 20. Objects can translate, rotate, collide, overlap, or leave the camera view. This creates several distinct demands:

- preserve object identity across frames;
- distinguish displacement from total path length;
- use the requested frame interval rather than the whole video;
- convert frames to elapsed time using 12 FPS;
- interpret motion relative to the camera when asked for direction;
- compare several candidate objects when asked for a maximum.

The source data is produced by the [MOVi-A export and build script](../kubric/pbs_run_movi_a_motion.sh), while the diagnostic conditions are defined in the [task utilities](lmms_eval/tasks/kubric_movi_a_motion/utils.py).

## The eight motion question types

| Question type | What the model must compute | Core evidence |
|---|---|---|
| Displacement | Euclidean distance between the start and end positions | Two endpoint positions |
| Path length | Sum of all consecutive 3D segment lengths | Full position trace over the interval |
| Average speed | Path length divided by elapsed time | Position trace, frame interval, FPS |
| Instantaneous speed | Magnitude of a velocity vector | Velocity near the query frame |
| Instantaneous acceleration | Magnitude of the change in velocity per unit time | Neighboring velocity vectors and FPS |
| Dominant motion direction | Largest signed change in camera-relative axes or camera distance | Camera-frame endpoint changes |
| Fastest object | Largest instantaneous speed among candidate objects | Candidate velocity vectors |
| Greatest path-length object | Largest accumulated path length among candidates | Candidate position traces |

The numerical questions use physically related distractors rather than arbitrary numbers. For example, options can correspond to displacement instead of path length, an incorrect time denominator, a component magnitude instead of a vector norm, or a rounded/intermediate value. This makes an incorrect option informative about the likely reasoning error.

## Baseline example

At baseline, the model sees the ordered video frames and the question, but no privileged ground-truth state.

```text
These are the ordered frames of a 12 FPS rigid-body simulation.

Question: What is the average speed of the large red metal cylinder
from frame 0 to frame 8?

Options:
A. 6.523 scene units/s
B. 4.892 scene units/s
C. 9.785 scene units/s
D. 8.154 scene units/s
E. 3.262 scene units/s

Answer with only the option letter A, B, C, D, or E.
```

To solve this correctly, the model must track the red cylinder through frames 0–8, estimate its 3D path rather than only its endpoint displacement, infer that eight frame intervals at 12 FPS equal `8/12 = 0.6667` seconds, and match the result to option A.

## How the diagnostics work

Each diagnostic condition changes one kind of evidence while preserving the source question. Conditions are grouped into eight suites.

| Diagnostic suite | Controlled intervention | What an improvement suggests |
|---|---|---|
| Oracle ladder | Progressively supplies timing, tracking, 2D motion, physical state, formulas, derived values, answer text, and finally the answer letter | The stage at which accuracy rises identifies the missing capability |
| Perception | Bounding boxes, segmentation overlays, visibility traces, object-center tracks, or uniform candidate tracks | Object localization, identity, or visibility was the bottleneck |
| Temporal | Frame labels, highlighted endpoints, query-centered triplets, interval cropping, collision timelines, or explicit FPS/Δt | Frame indexing or temporal selection was the bottleneck |
| Coordinate | World/camera position tables, camera basis, camera-distance traces, top-down trajectories, signed deltas, or direction definitions | Coordinate grounding or viewpoint interpretation was the bottleneck |
| Calculation | Formulas, raw inputs, vector differences, squared components, segment vectors/lengths, velocity differences, or derived values | Arithmetic or physical calculation was the bottleneck |
| Representation | The same task-relevant state expressed as a compact table, JSON, or trajectory plot | Performance depends on how equivalent evidence is encoded |
| Precision | State rounded to 2 decimals, 4 decimals, or near-full precision | Numerical precision materially affects the answer |
| Robustness | Noisy coordinates, reversed frames, irrelevant distractor state, or explicitly mismatched object state | The model is sensitive to noise, order, or misleading context |

### Perception and tracking intervention

The tracking scaffold marks the queried object while preserving the underlying scene. The green and red borders identify the query interval endpoints.

![Tracking overlay for frames 0, 4, and 8](docs/assets/kubric_movi_a_motion/sequence_1_tracking_query.png)

If this intervention improves accuracy over the paired baseline, the original error is more consistent with locating or tracking the target than with arithmetic.

### Coordinate intervention

Some questions are easier to express in a world-space trajectory than in perspective images. The trajectory scaffold supplies the same queried motion viewed from above.

![Top-down world-space trajectory of object 1](docs/assets/kubric_movi_a_motion/sequence_1_top_down_trajectory.png)

This is especially useful for separating visual tracking from reasoning over a recovered trajectory.

## Example assisted prompts

### Physical state plus formula

This condition removes most perception uncertainty but still requires the model to carry out the calculation.

```text
These are the ordered frames of a 12 FPS rigid-body simulation.

Authoritative GT physical-state scaffold (scene units):
large red metal cylinder positions:
f0=[-0.1669, 2.7795, 2.3529];
f1=[-0.3809, 2.2744, 2.2359];
f2=[-0.4809, 1.7797, 2.0050];
f3=[-0.5808, 1.2850, 1.7046];
f4=[-0.6808, 0.7903, 1.3348];
f5=[-0.7808, 0.2956, 0.8956];
f6=[-0.8218, -0.1745, 0.7628];
f7=[-0.8767, -0.5968, 0.8541];
f8=[-0.9323, -1.0165, 0.8915]

Computation scaffold: average speed = path length / elapsed time

Question: What is the average speed of the large red metal cylinder
from frame 0 to frame 8?

Options:
A. 6.523 scene units/s
B. 4.892 scene units/s
C. 9.785 scene units/s
D. 8.154 scene units/s
E. 3.262 scene units/s

Answer with only the option letter A, B, C, D, or E.
```

Interpretation:

- If baseline fails but this succeeds, perception/tracking or formula recall was probably limiting.
- If this also fails but a derived-value oracle succeeds, the remaining issue is calculation or option matching.
- If even the answer-letter oracle fails, the problem is likely instruction following or response parsing.

### Intermediate calculation scaffold

This condition supplies consecutive segment lengths but leaves the final summation and time division to the model.

```text
These are the ordered frames of a 12 FPS rigid-body simulation.

GT consecutive segment lengths: object 1:
[0.560947, 0.555002, 0.587300, 0.625667,
 0.669061, 0.490291, 0.435519, 0.425068]

Question: What is the average speed of the large red metal cylinder
from frame 0 to frame 8?

Options:
A. 6.523 scene units/s
B. 4.892 scene units/s
C. 9.785 scene units/s
D. 8.154 scene units/s
E. 3.262 scene units/s

Answer with only the option letter A, B, C, D, or E.
```

This intervention is more diagnostic than directly supplying the final value: it asks whether the model can finish a short, well-specified numerical computation once perception has been removed.

### Coordinate-direction scaffold

```text
These are the ordered frames of a 12 FPS rigid-body simulation.

GT signed motion components:
camera_delta=[-0.9183, -0.2184, 0.3465],
distance_to_camera_delta=0.0460.

Direction rule: positive camera-X=left, negative camera-X=right,
positive camera-Y=up, negative camera-Y=down,
decreasing camera distance=toward, increasing distance=away.

Question: What is the dominant motion direction of the large brown
metal sphere from frame 16 to frame 20?

Options:
A. toward the camera
B. left
C. down
D. away from the camera
E. right

Answer with only the option letter A, B, C, D, or E.
```

This tests whether errors come from estimating motion versus interpreting the camera convention.

### Candidate-comparison scaffold

```text
These are the ordered frames of a 12 FPS rigid-body simulation.

Authoritative GT physical-state scaffold (scene units):
small blue rubber cylinder, frame 6: velocity=[-1.9203, -0.3818, -5.8333]
large brown metal sphere, frame 6: velocity=[-1.2055, -4.3870, 0.0831]
large yellow rubber sphere, frame 6: velocity=[-0.9436, -6.3779, -3.3877]
large red metal cylinder, frame 6: velocity=[-0.4929, -5.6420, -1.9898]
small gray rubber sphere, frame 6: velocity=[-5.0207, -1.7387, -5.8333]

Question: Which of these objects is moving fastest at frame 6?

Options:
A. small blue rubber cylinder
B. large brown metal sphere
C. large yellow rubber sphere
D. large red metal cylinder
E. small gray rubber sphere

Answer with only the option letter A, B, C, D, or E.
```

The model must compute each velocity norm and select the maximum. The scaffold eliminates object tracking but does not directly reveal the winner.

### Representation comparison

The representation suite expresses the same task-relevant fields and frame rows in different forms. For example, the compact table condition begins:

```text
Authoritative task-relevant GT table (scene units):
object | frame | time | position_world
1 | 0 | 0.0         | [-0.166878,2.77953,2.35292]
1 | 1 | 0.083333333 | [-0.380929,2.2744,2.23591]
1 | 2 | 0.166666667 | [-0.480886,1.77971,2.005]
...
1 | 8 | 0.666666667 | [-0.932295,-1.01655,0.891535]
```

The JSON condition carries equivalent values:

```json
{
  "1": [
    {"frame": 0, "position_world": [-0.166878, 2.779527, 2.352925], "time": 0.0},
    {"frame": 1, "position_world": [-0.380929, 2.274402, 2.23591], "time": 0.083333333},
    {"frame": 2, "position_world": [-0.480886, 1.779713, 2.005], "time": 0.166666667}
  ]
}
```

A table-versus-JSON difference therefore primarily measures representation sensitivity rather than access to different physical information. The table uses compact significant-digit formatting while JSON preserves the task payload's rounded decimal values, so a small formatting/precision difference remains and should be acknowledged when interpreting a narrow gap.

### Robustness control

The mismatched-object condition explicitly supplies state from the wrong object:

```text
INTENTIONALLY MISMATCHED CONTROL: the following belongs to a different
object and must not be used as target evidence. large brown metal sphere
positions: f0=[1.8804, 1.3942, 1.9832]; ...; f8=[1.0390, -1.7984, 0.7023]

Question: What is the average speed of the large red metal cylinder
from frame 0 to frame 8?
```

This is not intended to help. It measures whether the model follows the explicit warning and relies on the video rather than blindly trusting salient numerical context.

## Metrics and comparisons

The evaluation records:

- `accuracy`: whether the returned option letter matches the gold answer;
- `baseline_accuracy`: accuracy on the unassisted condition;
- `assisted_accuracy`: accuracy under a diagnostic scaffold;
- `paired_scaffold_gain`: within-question improvement relative to baseline;
- `parse_success`: whether the response contains a valid option letter;
- `numeric_option_error`: numerical distance associated with the selected option, where applicable.

Paired scaffold gain is the key diagnostic quantity. Comparing unrelated baseline and assisted questions would confound question difficulty; pairing conditions by source question avoids that problem.

## How to interpret common result patterns

| Result pattern | Likely interpretation |
|---|---|
| Bounding boxes or center tracks help | Target localization or identity tracking is weak |
| Interval crops or endpoint highlights help | The model uses irrelevant frames or misreads temporal bounds |
| FPS/Δt helps | Frame-to-time conversion is weak |
| Raw state helps, formula-only does not | Perception/state estimation dominates the error |
| Formula-only helps, raw state does not | The model can estimate state but forgets or misapplies the relation |
| Segment lengths help more than positions | Vector differencing or norm calculation is weak |
| Derived quantity helps but intermediate scaffolds do not | Final arithmetic or multistep accumulation is weak |
| Camera-direction definition helps | Coordinate-sign convention is weak |
| Table strongly outperforms JSON | Structured-data representation affects evidence use |
| Full precision outperforms 2-decimal state | The options require precision beyond coarse estimates |
| Reversing frames causes a large drop | The model is sensitive to temporal order, as desired for signed motion |
| Mismatched state causes a large drop | The model over-trusts conflicting text over visual evidence |
| Answer-letter oracle has poor parse success | Instruction following or output formatting is broken |

## Important caveats

1. **Oracle accuracy is not ordinary video understanding.** Privileged state deliberately removes parts of the problem. Oracle conditions identify bottlenecks and upper bounds; they should not be reported as baseline capability.

2. **The final oracle steps intentionally leak the answer.** Answer-text and answer-letter oracles verify whether the remaining failure is option mapping, instruction following, or parsing.

3. **Synthetic ground truth makes decomposition possible.** Exact coordinates and velocities are available because the scenes are simulated. Real-video performance may include additional errors from imperfect annotations and camera estimation.

4. **Path length and displacement are different.** A model can correctly locate both endpoints and still fail if it substitutes straight-line displacement for the traveled path.

5. **Repeated baselines are paired controls.** The diagnostic group repeats the same source questions across suites so each scaffold can be compared with its own baseline.

6. **Accuracy alone can hide near misses.** Parse success and numeric option error help distinguish formatting failures from physically plausible but numerically incorrect reasoning.

## Experiment scale

The generated dataset contains 232,834 multiple-choice records. The diagnostic evaluation samples 100 source questions from each of the eight motion subtasks, giving 800 unique source questions. Expanding those questions across applicable scaffold conditions produces 40,900 model requests in the combined diagnostic group.

The group definition is in [kubric_movi_a_motion_diagnostics.yaml](lmms_eval/tasks/kubric_movi_a_motion/kubric_movi_a_motion_diagnostics.yaml), and the common task template and metrics are in [_default_template_yaml](lmms_eval/tasks/kubric_movi_a_motion/_default_template_yaml).

## One-sentence summary

This experiment uses paired interventions on synthetic motion questions to determine whether a vision-language model's errors arise from perception, tracking, temporal grounding, coordinate interpretation, physical calculation, data representation, numerical precision, or robustness—not merely whether its final multiple-choice answer is correct.
