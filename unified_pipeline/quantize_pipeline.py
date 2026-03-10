#!/usr/bin/env python3
"""
===============================================================================
QualViz INT8 Quantization Pipeline
===============================================================================
Supports: segmentation | classification | anomaly (PatchCore only)
All quantization uses QDQ format.

Steps run automatically:
  1. Inspect        — inputs, outputs, opset, node breakdown
  2. Find nodes     — scan model to find all incompatible nodes to exclude
  3. Upgrade opset  — upgrade to opset 17 if needed (fixes DequantizeLinear error)
  4. Preprocess     — shape inference (required before calibration)
  5. Quantize       — QDQ INT8 static quantization
  6. Validate       — benchmark FP32 vs INT8, check decision match

Usage:
    python3 quantize_pipeline.py --model_path /path/to/model.onnx
                                 --model_type segmentation
                                 --calib_dir  /path/to/calib_images/

    python3 quantize_pipeline.py --model_path /path/to/model.onnx
                                 --model_type classification
                                 --calib_dir  /path/to/calib_images/

    python3 quantize_pipeline.py --model_path /path/to/model.onnx
                                 --model_type anomaly
                                 --calib_dir  /path/to/calib_images/anomaly/
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
from onnx import version_converter
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quant_pre_process,
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="QualViz INT8 Quantization Pipeline")
    parser.add_argument("--model_path",  required=True,
                        help="Path to input FP32 ONNX model")
    parser.add_argument("--model_type",  required=True,
                        choices=["segmentation", "classification", "anomaly"],
                        help="Model type")
    parser.add_argument("--calib_dir",   required=True,
                        help="Path to calibration images folder")
    parser.add_argument("--output_dir",  default=None,
                        help="Output directory (default: same as model)")
    parser.add_argument("--input_size",  default=None, type=int,
                        help="Input size override e.g. 256 or 512")
    parser.add_argument("--num_threads", default=3, type=int,
                        help="ORT intra_op threads (default: 3)")
    parser.add_argument("--target_opset", default=17, type=int,
                        help="Target opset for upgrade (default: 17)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — ORT version check
# ─────────────────────────────────────────────────────────────────────────────

def check_ort_version():
    print("\n" + "="*60)
    print("STEP 1 — Environment Check")
    print("="*60)

    ort_ver = ort.__version__
    onnx_ver = onnx.__version__
    print(f"ORT version  : {ort_ver}")
    print(f"ONNX version : {onnx_ver}")

    # ORT 1.14+ required for QDQ per_channel on ARM
    major, minor = int(ort_ver.split(".")[0]), int(ort_ver.split(".")[1])
    if major < 1 or (major == 1 and minor < 14):
        print(f"\nWARNING: ORT {ort_ver} is too old for QDQ per_channel.")
        print("         Please upgrade: pip install --upgrade onnxruntime")
        print("         Minimum required: 1.14.0")
        sys.exit(1)
    else:
        print(f"ORT version OK (>= 1.14.0 required for QDQ per_channel)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Inspect model
# ─────────────────────────────────────────────────────────────────────────────

def inspect_model(model_path, model_type):
    print("\n" + "="*60)
    print("STEP 2 — Model Inspection")
    print("="*60)

    model = onnx.load(model_path)
    sess  = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    opset = model.opset_import[0].version
    print(f"Opset        : {opset}")
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

    op_counts = {}
    for n in model.graph.node:
        op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1

    print("\nTop op types:")
    for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {op:30s}: {cnt}")

    # ── Modification 1 — Auto-detect input format (NCHW / NHWC / grayscale) ──
    input_shape   = inputs[0].shape
    detected_size = None
    channels      = 3          # default
    layout        = "NCHW"    # default

    if len(input_shape) == 4:
        dim1 = input_shape[1]
        dim2 = input_shape[2]
        dim3 = input_shape[3]

        # NCHW: [1, C, H, W]
        if isinstance(dim1, int) and dim1 in (1, 3, 4):
            channels = dim1
            if isinstance(dim2, int) and dim2 > 0:
                detected_size = dim2
            layout = "NCHW"

        # NHWC: [1, H, W, C]
        elif isinstance(dim3, int) and dim3 in (1, 3, 4):
            channels = dim3
            if isinstance(dim2, int) and dim2 > 0:
                detected_size = dim2
            layout = "NHWC"

        # fallback — assume NCHW with 3 channels
        else:
            if isinstance(dim2, int) and dim2 > 0:
                detected_size = dim2

    print(f"\nDetected layout   : {layout}")
    print(f"Detected channels : {channels}")
    if detected_size:
        print(f"Detected input size: {detected_size}")

    if layout == "NHWC":
        print("WARNING: Model uses NHWC layout [1,H,W,C].")
        print("         Calibration reader will transpose accordingly.")

    if channels == 1:
        print("NOTE: Grayscale model detected (1 channel).")
        print("      Calibration reader will convert images to grayscale.")

    # ── Modification 2 — Multi-input detection ────────────────────────────────
    if len(inputs) > 1:
        print(f"\nWARNING: Model has {len(inputs)} inputs:")
        for inp in inputs:
            print(f"  {inp.name}  {inp.shape}  {inp.type}")
        print("  Pipeline will feed calibration data to ALL inputs.")
        print("  If inputs have different meanings (e.g. depth map), results may be inaccurate.")

    return opset, detected_size, inputs[0].name, channels, layout, inputs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Find incompatible nodes (scan actual model)
# ─────────────────────────────────────────────────────────────────────────────

def find_nodes_to_exclude(model_path, model_type):
    print("\n" + "="*60)
    print("STEP 3 — Find Incompatible Nodes")
    print("="*60)

    # Ops that are NOT INT8-compatible on ORT ARM builds
    problematic_ops = {
        "segmentation":   [
            "ReduceMax", "ReduceMean", "ReduceMin", "ReduceSum", "ReduceProd",
            "Resize", "Upsample", "Softmax", "Sigmoid", "LogSoftmax",
            "ArgMax", "ArgMin", "TopK", "NonMaxSuppression",
            "ScatterND", "GatherND", "GridSample",
        ],
        "classification": [
            "Softmax", "Sigmoid", "LogSoftmax",
            "ArgMax", "ArgMin", "TopK",
            "ReduceMax", "ReduceMean", "Resize",
        ],
        "anomaly":        [
            "ReduceMax", "ReduceMean", "ReduceMin", "ReduceSum", "ReduceProd",
            "Resize", "Upsample", "Softmax", "Sigmoid", "LogSoftmax",
            "ArgMax", "ArgMin", "TopK", "NonMaxSuppression",
            "GlobalAveragePool", "Clip", "Greater", "MatMul",
        ],
    }

    model      = onnx.load(model_path)
    excl_ops   = problematic_ops[model_type]
    found      = []

    print(f"Scanning {len(model.graph.node)} nodes for incompatible ops...")
    print(f"Checking ops: {excl_ops}\n")

    for node in model.graph.node:
        if node.op_type in excl_ops:
            print(f"  [EXCLUDE] op={node.op_type:<25} name={node.name}")
            found.append(node.name)

    print(f"\nTotal nodes to exclude: {len(found)}")

    # ── Modification 3 — Conv ratio warning ───────────────────────────────────
    op_counts  = {}
    for n in model.graph.node:
        op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1
    conv_count  = op_counts.get("Conv", 0)
    total_nodes = len(model.graph.node)
    conv_ratio  = conv_count / total_nodes if total_nodes > 0 else 0

    print(f"\nConv ratio : {conv_count} / {total_nodes} = {conv_ratio*100:.0f}%")
    if conv_ratio <= 0.10:
        print("WARNING: 10% or fewer Conv nodes detected.")
        print("         INT8 quantization is UNLIKELY to speed up this model.")
        print("         Expected speedup: 0.7x–1.1x (may be slower than FP32).")
        print("         Recommendation: consider keeping FP32 for this model.")
    elif conv_ratio < 0.25:
        print("NOTE: Low Conv ratio (10–25%). Modest speedup expected (~1.2x–1.5x).")
    else:
        print(f"Good Conv ratio (>25%). Meaningful INT8 speedup expected.")

    return found


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Upgrade opset if needed
# ─────────────────────────────────────────────────────────────────────────────

def upgrade_opset_if_needed(model_path, output_dir, current_opset, target_opset):
    print("\n" + "="*60)
    print("STEP 4 — Opset Check & Upgrade")
    print("="*60)
    print(f"Current opset : {current_opset}")
    print(f"Target opset  : {target_opset}")

    if current_opset >= target_opset:
        print(f"Opset is already {current_opset}. No upgrade needed.")
        return model_path

    print(f"Upgrading opset {current_opset} → {target_opset}...")
    stem        = Path(model_path).stem
    opset_path  = os.path.join(output_dir, f"{stem}_opset{target_opset}.onnx")

    model = onnx.load(model_path)

    try:
        # Method 1 — automatic conversion (preferred)
        converted = version_converter.convert_version(model, target_opset)
        onnx.save(converted, opset_path)
        upgraded  = onnx.load(opset_path)
        onnx.checker.check_model(upgraded)
        print(f"Opset upgraded successfully → {upgraded.opset_import[0].version}")
        print(f"Saved to: {opset_path}")

    except Exception as e:
        print(f"Auto conversion failed: {e}")
        print("Trying manual opset bump (fallback)...")

        # Method 2 — manual bump
        try:
            del model.opset_import[:]
            opset_entry         = model.opset_import.add()
            opset_entry.domain  = ""
            opset_entry.version = target_opset
            onnx.save(model, opset_path)
            print(f"Manual opset bump done → {target_opset}")
            print(f"Saved to: {opset_path}")

            try:
                onnx.checker.check_model(onnx.load(opset_path))
                print("Model validation passed!")
            except Exception as ve:
                print(f"Validation warning (may still work): {ve}")

        except Exception as e2:
            print(f"Both opset upgrade methods failed: {e2}")
            print("Continuing with original model opset. May fail at quantization.")
            return model_path

    return opset_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Shape inference preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_model(model_path, output_dir):
    print("\n" + "="*60)
    print("STEP 5 — Shape Inference Preprocessing")
    print("="*60)

    stem      = Path(model_path).stem
    prep_path = os.path.join(output_dir, f"{stem}_preprocessed.onnx")

    print(f"Input  : {model_path}")
    print(f"Output : {prep_path}")

    quant_pre_process(
        input_model_path=model_path,
        output_model_path=prep_path,
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
        auto_merge=True,
        int_max=2**31 - 1,
        verbose=1,
    )

    print(f"Preprocessed size: {os.path.getsize(prep_path)/1024/1024:.1f} MB")
    return prep_path


# ─────────────────────────────────────────────────────────────────────────────
# Calibration data reader
# ─────────────────────────────────────────────────────────────────────────────

class UniversalCalibReader(CalibrationDataReader):
    """
    Handles:
      - NCHW layout  [1, 3, H, W]  (standard)
      - NHWC layout  [1, H, W, 3]  (some TF-exported models)
      - Grayscale    [1, 1, H, W]
      - Multi-input  (feeds same image data to all inputs)
    """
    def __init__(self, calib_dir, all_inputs, input_size,
                 channels=3, layout="NCHW"):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.imgs = sorted([
            str(p) for p in Path(calib_dir).iterdir()
            if p.suffix.lower() in exts
        ])
        if not self.imgs:
            raise FileNotFoundError(f"No images found in {calib_dir}")

        self.all_inputs  = all_inputs   # list of ORT input metadata
        self.input_size  = input_size
        self.channels    = channels
        self.layout      = layout
        self.idx         = 0
        print(f"Calibration images found : {len(self.imgs)}")
        print(f"Input layout             : {layout}")
        print(f"Input channels           : {channels}")
        if len(all_inputs) > 1:
            print(f"Multi-input model        : {len(all_inputs)} inputs")

    def _prepare_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None

        # Grayscale handling
        if self.channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.input_size, self.input_size))
            img = img.astype(np.float32) / 255.0
            img = img[np.newaxis, np.newaxis, :, :]   # [1,1,H,W]
            if self.layout == "NHWC":
                img = img.transpose(0, 2, 3, 1)        # [1,H,W,1]
            return img

        # BGR -> handle 4-channel PNG
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        img = cv2.resize(img, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        if self.layout == "NCHW":
            img = np.transpose(img, (2, 0, 1))         # HWC -> CHW
            img = np.expand_dims(img, axis=0)           # [1,3,H,W]
        else:  # NHWC
            img = np.expand_dims(img, axis=0)           # [1,H,W,3]

        return img

    def get_next(self):
        if self.idx >= len(self.imgs):
            return None

        img_path = self.imgs[self.idx]
        self.idx += 1
        print(f"  Calibrating [{self.idx}/{len(self.imgs)}]")

        img = self._prepare_image(img_path)
        if img is None:
            return self.get_next()

        # Feed to all inputs — multi-input models get same image for all
        feed = {}
        for inp in self.all_inputs:
            feed[inp.name] = img

        return feed


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Quantization
# ─────────────────────────────────────────────────────────────────────────────

def quantize_model(prep_path, output_dir, model_type,
                   calib_dir, all_inputs, input_name,
                   input_size, nodes_to_excl, channels, layout):
    print("\n" + "="*60)
    print("STEP 6 — INT8 Static Quantization (QDQ)")
    print("="*60)

    stem     = Path(prep_path).stem
    # Remove intermediate suffixes from output name
    for suffix in ["_preprocessed", f"_opset{17}", "_opset17", "_opset13"]:
        stem = stem.replace(suffix, "")
    out_path = os.path.join(output_dir, f"{stem}_int8_static.onnx")

    print(f"Format       : QDQ")
    print(f"per_channel  : True")
    print(f"reduce_range : True")
    print(f"weight_type  : QInt8")
    print(f"Output       : {out_path}")

    reader = UniversalCalibReader(calib_dir, all_inputs, input_size,
                                  channels=channels, layout=layout)

    quantize_static(
        model_input             = prep_path,
        model_output            = out_path,
        calibration_data_reader = reader,
        quant_format            = QuantFormat.QDQ,
        weight_type             = QuantType.QInt8,
        activation_type         = QuantType.QInt8,
        per_channel             = True,
        reduce_range            = True,
        nodes_to_exclude        = nodes_to_excl,
    )

    print(f"\nQuantized model size: {os.path.getsize(out_path)/1024/1024:.1f} MB")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Validation & Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def validate_models(fp32_path, int8_path, input_name,
                    input_size, model_type, num_threads,
                    channels=3, layout="NCHW"):
    print("\n" + "="*60)
    print("STEP 7 — Validation & Benchmark")
    print("="*60)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.add_session_config_entry(
        "session.memory.enable_memory_arena_shrinkage", "cpu:0")

    fp32_sess = ort.InferenceSession(fp32_path,  opts,
                                     providers=["CPUExecutionProvider"])
    int8_sess = ort.InferenceSession(int8_path,  opts,
                                     providers=["CPUExecutionProvider"])

    # Build dummy with correct shape for this model's layout and channel count
    if layout == "NCHW":
        dummy = np.random.randn(1, channels, input_size, input_size).astype(np.float32)
    else:  # NHWC
        dummy = np.random.randn(1, input_size, input_size, channels).astype(np.float32)
    print(f"Dummy input shape: {dummy.shape}")

    # Warmup — 2 runs each
    fp32_sess.run(None, {input_name: dummy})
    fp32_sess.run(None, {input_name: dummy})
    int8_sess.run(None, {input_name: dummy})
    int8_sess.run(None, {input_name: dummy})

    # Benchmark — 5 timed runs
    fp32_times, int8_times = [], []
    for _ in range(5):
        t = time.time()
        fp32_out = fp32_sess.run(None, {input_name: dummy})
        fp32_times.append((time.time() - t) * 1000)

        t = time.time()
        int8_out = int8_sess.run(None, {input_name: dummy})
        int8_times.append((time.time() - t) * 1000)

    fp32_avg = np.mean(fp32_times)
    int8_avg = np.mean(int8_times)
    speedup  = fp32_avg / int8_avg

    print(f"FP32  avg : {fp32_avg:.1f} ms")
    print(f"INT8  avg : {int8_avg:.1f} ms")
    print(f"Speedup   : {speedup:.2f}x")

    # Decision match
    print(f"\nOutput count — FP32: {len(fp32_out)}  INT8: {len(int8_out)}")

    if model_type in ["segmentation", "classification"]:
        fp32_pred = int(np.argmax(fp32_out[0].flatten()))
        int8_pred = int(np.argmax(int8_out[0].flatten()))
        match     = fp32_pred == int8_pred
        print(f"FP32 argmax: {fp32_pred}  |  INT8 argmax: {int8_pred}")
        print(f"Decision match: {match}")

    elif model_type == "anomaly":
        fp32_score = float(np.array(fp32_out[0]).flatten()[0])
        int8_score = float(np.array(int8_out[0]).flatten()[0])
        fp32_anom  = fp32_score > 0.5
        int8_anom  = int8_score > 0.5
        match      = fp32_anom == int8_anom
        print(f"FP32 score={fp32_score:.4f} → {'ANOMALY' if fp32_anom else 'OK'}")
        print(f"INT8 score={int8_score:.4f} → {'ANOMALY' if int8_anom else 'OK'}")
        print(f"Decision match: {match}")

    # Node count sanity check
    int8_model    = onnx.load(int8_path)
    qdq_q         = sum(1 for n in int8_model.graph.node if n.op_type == "QuantizeLinear")
    conv_fp32     = sum(1 for n in int8_model.graph.node if n.op_type == "Conv")
    print(f"\nNode check:")
    print(f"  QuantizeLinear nodes : {qdq_q}")
    print(f"  FP32 Conv remaining  : {conv_fp32}")
    if qdq_q == 0:
        print("  WARNING: QuantizeLinear=0 — quantization may have failed!")
    else:
        print("  Quantization looks correct.")

    return fp32_avg, int8_avg, speedup


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(model_path, int8_path, model_type,
                  input_size, fp32_avg, int8_avg, speedup):
    print("\n" + "="*60)
    print("STEP 8 — Summary")
    print("="*60)
    orig_mb = os.path.getsize(model_path) / 1024 / 1024
    int8_mb = os.path.getsize(int8_path)  / 1024 / 1024
    print(f"Model type   : {model_type}")
    print(f"Input size   : {input_size}x{input_size}")
    print(f"FP32 model   : {os.path.basename(model_path)}  ({orig_mb:.1f} MB)")
    print(f"INT8 model   : {os.path.basename(int8_path)}   ({int8_mb:.1f} MB)")
    print(f"Size reduction: {((orig_mb - int8_mb) / orig_mb) * 100:.1f}%")
    print(f"FP32 time    : {fp32_avg:.1f} ms")
    print(f"INT8 time    : {int8_avg:.1f} ms")
    print(f"Speedup      : {speedup:.2f}x")
    if speedup < 1.0:
        print(f"\nWARNING: INT8 is slower than FP32 for this model.")
        print("         Too few Conv nodes to benefit from quantization.")
        print("         Recommendation: keep FP32 for inference.")
    else:
        print(f"\nDeploy this model: {int8_path}")
    print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

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

    # Step 1 — ORT + ONNX version check
    check_ort_version()

    # Step 2 — Inspect
    current_opset, detected_size, input_name, channels, layout, all_inputs = inspect_model(
        args.model_path, args.model_type)

    # Resolve input size
    input_size = args.input_size or detected_size
    if input_size is None:
        fname      = Path(args.model_path).stem
        digit_pos  = next((i for i, c in enumerate(fname) if c.isdigit()), None)
        if digit_pos is not None:
            try:
                input_size = int(fname[digit_pos:digit_pos + 3])
            except Exception:
                input_size = 256
        else:
            input_size = 256
    print(f"\nUsing input_size: {input_size}")

    # Step 3 — Find incompatible nodes by scanning actual model
    nodes_to_excl = find_nodes_to_exclude(args.model_path, args.model_type)

    # Step 4 — Upgrade opset if needed
    upgraded_path = upgrade_opset_if_needed(
        args.model_path, args.output_dir,
        current_opset, args.target_opset)

    # Step 5 — Shape inference preprocessing
    prep_path = preprocess_model(upgraded_path, args.output_dir)

    # Step 6 — Quantize
    int8_path = quantize_model(
        prep_path, args.output_dir, args.model_type,
        args.calib_dir, all_inputs, input_name,
        input_size, nodes_to_excl, channels, layout)

    # Step 7 — Validate & benchmark
    fp32_avg, int8_avg, speedup = validate_models(
        args.model_path, int8_path, input_name,
        input_size, args.model_type, args.num_threads,
        channels=channels, layout=layout)

    # Step 8 — Summary
    print_summary(args.model_path, int8_path, args.model_type,
                  input_size, fp32_avg, int8_avg, speedup)


if __name__ == "__main__":
    main()
