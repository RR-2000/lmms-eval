# ScanNet camera-basis direction prediction

This task evaluates camera-relative 3D spatial reasoning on prepared ScanNet
v2 RGB frames under `/mnt/rdata4_3/Spatial_Benchmarks/scannetv2`.

ScanNet preparation supplies instance masks, 2D boxes, camera poses, and
object 3D centres. It does **not** supply a dependable semantic front/right
orientation for each reconstructed object. The object-basis tasks therefore
use the explicit, reproducible anchor-facing-camera construction below, rather
than claiming that the reconstructed mesh has a known semantic yaw.

## Coordinate frame and ground truth

For each selected RGB frame, the manifest transforms each visible instance's
3D bounding-box centre into that frame's camera coordinates. The convention is
`+X` image-right, `+Y` camera forward, `+Z` image-up. For an ordered reference
and target pair, ground truth is the normalized target-minus-reference vector,
stored as JSON keys in the order `front`, `up`, `right`. The dominant absolute
horizontal component produces `left`, `right`, `front`, or `back`.

Each manifest view retains four to five visible non-structural instances whose
labels are unique within the entire frame. If a frame shows two objects with
the same label (for example, two chairs), neither is retained: prompts never
rely on an arbitrary instance number to identify an object. Floors, walls,
ceilings, doors, windows, and windowsills are excluded. The remaining objects
retain normalized 2D boxes and unique names for provenance. Frames are sampled
evenly per prepared scene to avoid adjacent video frames becoming near-duplicate
evaluation examples.

## Object basis

`scannet_object_basis_*` is an object-centric counterpart built from each
reference object's position. Imagine the reference object turning to look back
at the camera: the reference centre is the origin, horizontal anchor-to-camera
is `+front`, world up is `+up`, and `right = front × up`. The target-minus-
reference displacement is projected into this frame. Thus the object's right
appears camera/image-left. This construction supports an unambiguous object
perspective for ScanNet without inventing missing semantic object orientations.

## Build or refresh the manifest

Run after `scannet_download.sh` has prepared further scenes:

```bash
python lmms_eval/tasks/scannet_camera_basis/build_manifest.py --frames-per-scene 8
```

This writes `scannet_camera_basis_manifest.jsonl` beside the prepared ScanNet
scenes and prints selected-view and ordered-pair counts for the currently
prepared data.

## Tasks

```text
scannet_camera_basis_direction
scannet_camera_basis_vector
scannet_camera_basis_combined
scannet_camera_basis
scannet_object_basis_direction
scannet_object_basis_vector
scannet_object_basis_combined
scannet_object_basis
scannet_camera_basis_object_direction
scannet_object_basis_object_direction
scannet_basis_object_direction
scannet_basis_all
```

The `*_camera_basis` and `*_object_basis` names run their respective direction,
vector, combined, and paired object/direction formats. `scannet_basis_all` runs
all eight. They use the same output schemas and metrics as the corresponding
COMFORT tasks.

## Paired object-answer versus direction-answer task

The two `*_basis_object_direction` tasks express every retained spatial fact
in inverse forms over the same image and geometry:

```text
From the camera's frame of reference, where is the chair relative to the table?
Return exactly one lowercase direction word: left, right, front, or back.
```

```text
From the camera's frame of reference, which candidate object is left of the table?
Return exactly one object name from this four-object candidate list:
chair, sofa, cabinet, lamp.
```

The object-facing-camera variant uses the constructed anchor frame described
above instead of camera coordinates. Both variants impose the following
controls so their aggregate object/direction comparison is meaningful:

- only five-object views are used, leaving exactly four candidates around each
  reference and matching the four direction labels;
- relations shared by multiple candidate objects are removed, because the
  inverse object question would otherwise have multiple correct answers;
- left, right, front, and back are deterministically downsampled to equal
  counts;
- direction and object rows share a `pair_id`, permitting paired outcome and
  scene-clustered statistical analysis.

With the current 980-view manifest, these controls yield 1,988 camera-frame
facts (3,976 rows) and 1,740 object-facing-camera facts (3,480 rows), exactly
balanced over the four relations. Rebuilding the manifest can change these
counts.

Run both coordinate frames with:

```bash
python -m lmms_eval \
  --model qwen3_vl_experiments \
  --model_args max_num_frames=32,pretrained="Qwen/Qwen3-VL-8B-Instruct" \
  --tasks scannet_basis_object_direction \
  --batch_size 1 --log_samples \
  --output_path outputs/scannet_basis_object_direction_8
```

The task reports overall, object-answer, direction-answer, parse, paired
transition, and object-minus-direction metrics. Submission records include the
prompt, four object candidates, gold answer, parsed prediction, frame, and
pair identifier.

```bash
python -m lmms_eval \
  --model qwen3_vl_experiments \
  --model_args max_num_frames=32,pretrained="Qwen/Qwen3-VL-4B-Instruct" \
  --tasks scannet_camera_basis \
  --batch_size 1 --log_samples \
  --output_path outputs/scannet_camera_basis/evaluation
```

Analyze all ScanNet basis submissions, including the paired tasks, with:

```bash
python tools/plot_scannet_basis_results.py \
  --inputs outputs/scannet_basis_all_8 outputs/scannet_basis_object_direction_8 \
  --output-dir outputs/scannet_basis_combined_analysis
```

In addition to the existing basis summaries, the analyzer writes
`object_vs_direction_accuracy.md`, `.csv`, and `.png`, plus
`object_direction_outcomes_100pct.png`.
