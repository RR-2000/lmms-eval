# COMFORT Multi 3D bounding-box prediction

`comfort_multi_3d_bbox_prediction` reads
`/home/ramanathan/data/COMFORT_Multi_3D/scenes.jsonl` and expands each scene
into one question per object. With the current dataset this produces 2,500
queries: 500 reference objects and 2,000 target objects.

The model must return exactly one normalized top-left-origin box:

```text
[x_min, y_min, x_max, y_max]
```

Coordinates use `[0,1000]`. The primary metric is mean IoU. The task also
reports IoU accuracy at 0.25, 0.5, and 0.75, parse success, and separate mean
IoU for reference and target objects.

## Debug overlays

Set `COMFORT_BBOX_DEBUG=1` to save a comparison image for every evaluated
query. Ground truth is green and the parsed prediction is red; the prediction
label includes its IoU. Parse failures retain the green GT box and show a red
failure message.

```bash
COMFORT_BBOX_DEBUG=1 \
COMFORT_BBOX_DEBUG_DIR=outputs/comfort_bbox_debug/overlays \
python -m lmms_eval \
  --model dummy --model_args 'response=[400 400 600 600]' \
  --tasks comfort_multi_3d_bbox_prediction \
  --limit 10 --batch_size 1 --log_samples \
  --output_path outputs/comfort_bbox_debug/evaluation
```
