#!/usr/bin/env python3
"""
===============================================================================
QualViz INT8 Quantization Pipeline
===============================================================================
Supports: segmentation | classification | anomaly
Usage:
    python3 quantize_pipeline.py --model_path /path/to/model.onnx
                                 --model_type segmentation
                                 --calib_dir  /path/to/calib_images/
                                 --output_dir /path/to/save/

    python3 quantize_pipeline.py --model_path /path/to/model.onnx
                                 --model_type classification
                                 --calib_dir  /path/to/calib_images/

    python3 quantize_pipeline.py --model_path /path/to/model.onnx
                                 --model_type anomaly
                                 --calib_dir  /path/to/calib_images/
===============================================================================
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2
import onnx
import onnxruntime as ort
from pathlib import Path

# ── ORT quantization imports ──────────────────────────────────────────────────
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    shape_inference as quant_shape_inference,
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="QualViz INT8 Quantization Pipeline")
    parser.add_argument("--model_path",  required=True,  help="Path to input FP32 ONNX model")
    parser.add_argument("--model_type",  required=True,
                        choices=["segmentation", "classification", "anomaly"],
                        help="Model type")
    parser.add_argument("--calib_dir",   required=True,  help="Path to calibration images folder")
    parser.add_argument("--output_dir",  default=None,   help="Output directory (default: same as model)")
    parser.add_argument("--input_size",  default=None,   type=int,
                        help="Input size override e.g. 256 or 512 (auto-detected from filename if not set)")
    parser.add_argument("--num_threads", default=3,      type=int, help="ORT intra_op threads (default: 3)")
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="Skip shape_inference preprocessing step")
    return parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Inspect model
# ─────────────────────────────────────────────────────────────────────────────

def inspect_model(model_path, model_type):
    print("\n" + "="*60)
    print("STEP 1 — Model Inspection")
    print("="*60)

    model = onnx.load(model_path)
    sess  = ort.InferenceSession(model_path)

    print(f"Opset        : {model.opset_import[0].version}")
    print(f"Model size   : {os.path.getsize(model_path)/1024/1024:.1f} MB")
    print(f"Total nodes  : {len(model.graph.node)}")

    inputs  = sess.get_inputs()
    outputs = sess.get_outputs()

    print("\nInputs:")
    for i in inputs:
        print(f"  {i.name}  {i.shape}  {i.type}")

    print("\nOutputs:")
    for o in outputs:
        print(f"  {o.name}  {o.shape}  {o.type}")

    # Count op types
    op_counts = {}
    for n in model.graph.node:
        op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1

    print("\nTop op types:")
    for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {op:30s}: {cnt}")

    # Auto-detect input size from shape
    input_shape = inputs[0].shape
    detected_size = None
    if len(input_shape) == 4:
        h = input_shape[2]
        w = input_shape[3]
        if isinstance(h, int) and h > 0:
            detected_size = h
            print(f"\nDetected input size from model: {h}x{w}")

    # Determine nodes to exclude per model type
    problematic_ops = {
        "segmentation" : ["Resize", "ArgMax", "TopK", "Softmax",
                          "NonMaxSuppression", "ScatterND", "GatherND"],
        "classification": ["Softmax", "ArgMax", "TopK", "Sigmoid"],
        "anomaly"       : ["Resize", "ArgMax", "TopK", "Softmax",
                           "Greater", "Clip", "NonMaxSuppression",
                           "GlobalAveragePool", "MatMul"],
    }
    exclude_ops   = problematic_ops[model_type]
    nodes_to_excl = [n.name for n in model.graph.node if n.op_type in exclude_ops]
    print(f"\nNodes to exclude ({len(nodes_to_excl)}): ops = {exclude_ops}")

    return detected_size, nodes_to_excl, inputs[0].name

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Preprocessing (shape inference)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_model(model_path, output_dir, skip=False):
    print("\n" + "="*60)
    print("STEP 2 — Shape Inference Preprocessing")
    print("="*60)

    stem = Path(model_path).stem
    prep_path = os.path.join(output_dir, f"{stem}_preprocessed.onnx")

    if skip:
        print("Skipped (--skip_preprocess). Using original model for calibration.")
        return model_path

    print(f"Input  : {model_path}")
    print(f"Output : {prep_path}")

    quant_shape_inference.quant_pre_process(
        model_path, prep_path, skip_optimization=False)

    print(f"Preprocessed size: {os.path.getsize(prep_path)/1024/1024:.1f} MB")
    return prep_path

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Calibration data reader
# ─────────────────────────────────────────────────────────────────────────────

class UniversalCalibReader(CalibrationDataReader):
    """Works for segmentation, classification, and anomaly models."""

    def __init__(self, calib_dir, input_name, input_size, channels=3):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.imgs = sorted([
            str(p) for p in Path(calib_dir).iterdir()
            if p.suffix.lower() in exts
        ])
        if not self.imgs:
            raise FileNotFoundError(f"No images found in {calib_dir}")

        self.input_name = input_name
        self.input_size = input_size
        self.channels   = channels
        self.idx        = 0
        print(f"Calibration images found: {len(self.imgs)}")

    def get_next(self):
        if self.idx >= len(self.imgs):
            return None

        img_path = self.imgs[self.idx]
        self.idx += 1

        img = cv2.imread(img_path)
        if img is None:
            return self.get_next()

        # Ensure BGR 3-channel
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        img = cv2.resize(img, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))       # HWC -> CHW
        img = np.expand_dims(img, axis=0)         # -> [1, 3, H, W]

        return {self.input_name: img}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Quantization
# ─────────────────────────────────────────────────────────────────────────────

def quantize_model(prep_path, output_dir, model_type,
                   calib_dir, input_name, input_size, nodes_to_excl):
    print("\n" + "="*60)
    print("STEP 4 — INT8 Static Quantization")
    print("="*60)

    stem      = Path(prep_path).stem.replace("_preprocessed", "")
    out_path  = os.path.join(output_dir, f"{stem}_int8_static.onnx")

    # Per model type settings — same as used in this project
    settings = {
        "segmentation":   dict(quant_format=QuantFormat.QDQ,
                               per_channel=True,
                               reduce_range=True,
                               weight_type=QuantType.QInt8),
        "classification": dict(quant_format=QuantFormat.QDQ,
                               per_channel=True,
                               reduce_range=True,
                               weight_type=QuantType.QInt8),
        "anomaly":         dict(quant_format=QuantFormat.QDQ,
                               per_channel=True,
                               reduce_range=True,
                               weight_type=QuantType.QInt8) ,
    }

    cfg = settings[model_type]
    print(f"Format       : {cfg['quant_format']}")
    print(f"per_channel  : {cfg['per_channel']}")
    print(f"reduce_range : {cfg['reduce_range']}")
    print(f"weight_type  : {cfg['weight_type']}")
    print(f"Output       : {out_path}")

    reader = UniversalCalibReader(calib_dir, input_name, input_size)

    quantize_static(
        model_input           = prep_path,
        model_output          = out_path,
        calibration_data_reader = reader,
        nodes_to_exclude      = nodes_to_excl,
        **cfg,
    )

    print(f"\nQuantized model size: {os.path.getsize(out_path)/1024/1024:.1f} MB")
    return out_path

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_models(fp32_path, int8_path, input_name, input_size,
                    model_type, num_threads):
    print("\n" + "="*60)
    print("STEP 5 — Validation & Benchmark")
    print("="*60)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    fp32_sess = ort.InferenceSession(fp32_path, opts)
    int8_sess = ort.InferenceSession(int8_path, opts)

    dummy = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

    # Warmup
    fp32_sess.run(None, {input_name: dummy})
    fp32_sess.run(None, {input_name: dummy})
    int8_sess.run(None, {input_name: dummy})
    int8_sess.run(None, {input_name: dummy})

    # Benchmark — 5 runs each
    fp32_times, int8_times = [], []
    for _ in range(5):
        t = time.time(); fp32_sess.run(None, {input_name: dummy})
        fp32_times.append((time.time()-t)*1000)
        t = time.time(); int8_sess.run(None, {input_name: dummy})
        int8_times.append((time.time()-t)*1000)

    fp32_avg = np.mean(fp32_times)
    int8_avg = np.mean(int8_times)
    speedup  = fp32_avg / int8_avg

    print(f"FP32  avg : {fp32_avg:.1f} ms")
    print(f"INT8  avg : {int8_avg:.1f} ms")
    print(f"Speedup   : {speedup:.2f}x")

    # Output name check per model type
    fp32_out = fp32_sess.run(None, {input_name: dummy})
    int8_out = int8_sess.run(None, {input_name: dummy})

    print(f"\nOutput count — FP32: {len(fp32_out)}  INT8: {len(int8_out)}")

    # Decision match for anomaly
    if model_type == "anomaly":
        fp32_score = float(fp32_out[0].flatten()[0])
        int8_score = float(int8_out[0].flatten()[0])
        fp32_anom  = bool(fp32_out[1].flatten()[0])
        int8_anom  = bool(int8_out[1].flatten()[0])
        print(f"FP32 score={fp32_score:.3f} anomaly={fp32_anom}")
        print(f"INT8 score={int8_score:.3f} anomaly={int8_anom}")
        print(f"Decision match: {fp32_anom == int8_anom}")

    # Node count check
    int8_model = onnx.load(int8_path)
    qlinear_conv = sum(1 for n in int8_model.graph.node if n.op_type == "QLinearConv")
    qdq_q        = sum(1 for n in int8_model.graph.node if n.op_type == "QuantizeLinear")
    conv_fp32    = sum(1 for n in int8_model.graph.node if n.op_type == "Conv")
    print(f"\nNode check:")
    print(f"  QLinearConv      : {qlinear_conv}")
    print(f"  QuantizeLinear   : {qdq_q}")
    print(f"  Conv (FP32 left) : {conv_fp32}")

    if model_type == "anomaly" and qlinear_conv == 0:
        print("  WARNING: QLinearConv=0 — Conv ops not quantized!")
    elif model_type in ["segmentation","classification"] and qdq_q == 0:
        print("  WARNING: QuantizeLinear=0 — quantization may have failed!")
    else:
        print("  Quantization looks correct.")

    return fp32_avg, int8_avg, speedup

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Summary report
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(model_path, int8_path, model_type,
                  input_size, fp32_avg, int8_avg, speedup):
    print("\n" + "="*60)
    print("STEP 6 — Summary")
    print("="*60)
    print(f"Model type   : {model_type}")
    print(f"Input size   : {input_size}x{input_size}")
    print(f"FP32 model   : {model_path}")
    print(f"  Size       : {os.path.getsize(model_path)/1024/1024:.1f} MB")
    print(f"INT8 model   : {int8_path}")
    print(f"  Size       : {os.path.getsize(int8_path)/1024/1024:.1f} MB")
    print(f"FP32 time    : {fp32_avg:.1f} ms")
    print(f"INT8 time    : {int8_avg:.1f} ms")
    print(f"Speedup      : {speedup:.2f}x")
    print(f"\nDeploy this model: {int8_path}")
    print("="*60)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve output dir
    if args.output_dir is None:
        args.output_dir = str(Path(args.model_path).parent)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("QualViz INT8 Quantization Pipeline")
    print("="*60)
    print(f"Model path  : {args.model_path}")
    print(f"Model type  : {args.model_type}")
    print(f"Calib dir   : {args.calib_dir}")
    print(f"Output dir  : {args.output_dir}")

    # Step 1 — inspect
    detected_size, nodes_to_excl, input_name = inspect_model(
        args.model_path, args.model_type)

    # Resolve input size
    input_size = args.input_size or detected_size
    if input_size is None:
        # fallback: parse from filename
        fname = Path(args.model_path).stem
        digit_pos = next((i for i, c in enumerate(fname) if c.isdigit()), None)
        if digit_pos is not None:
            try:
                input_size = int(fname[digit_pos:digit_pos+3])
            except:
                input_size = 256
        else:
            input_size = 256
    print(f"\nUsing input_size: {input_size}")

    # Step 2 — preprocess
    prep_path = preprocess_model(
        args.model_path, args.output_dir, skip=args.skip_preprocess)

    # Step 4 — quantize
    int8_path = quantize_model(
        prep_path, args.output_dir, args.model_type,
        args.calib_dir, input_name, input_size, nodes_to_excl)

    # Step 5 — validate
    fp32_avg, int8_avg, speedup = validate_models(
        args.model_path, int8_path, input_name,
        input_size, args.model_type, args.num_threads)

    # Step 6 — summary
    print_summary(args.model_path, int8_path, args.model_type,
                  input_size, fp32_avg, int8_avg, speedup)


if __name__ == "__main__":
    main()
