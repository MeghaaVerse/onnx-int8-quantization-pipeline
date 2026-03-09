# Unified Pipeline

Single script to quantize any model type via terminal arguments.

## Usage

```bash
source ~/onnx_env/bin/activate

# Segmentation
python3 quantize_pipeline.py \
    --model_path /path/to/seg_model.onnx \
    --model_type segmentation \
    --calib_dir  /path/to/calibration_images/

# Classification
python3 quantize_pipeline.py \
    --model_path /path/to/cls_model.onnx \
    --model_type classification \
    --calib_dir  /path/to/calibration_images/

# Anomaly (PatchCore only — not PaDiM)
python3 quantize_pipeline.py \
    --model_path /path/to/anomaly_model.onnx \
    --model_type anomaly \
    --calib_dir  /path/to/calibration_images/anomaly/
```

## What it does

1. **Inspect** — prints inputs, outputs, node type counts
2. **Preprocess** — shape inference (required before calibration)
3. **Calibrate** — runs images through model to collect activation stats
4. **Quantize** — converts to INT8
5. **Validate** — benchmarks FP32 vs INT8, checks decision match

## Output

Saves INT8 model as `<original_name>_int8_static.onnx` in the same directory.
