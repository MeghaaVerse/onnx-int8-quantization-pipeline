import onnxruntime as ort
import numpy as np
import cv2
import os
import time
import json
from pathlib import Path

# -------- CONFIG --------
MODEL_FP32        = "/home/qualviz/QualViz/models/Classification/512_2_clstalbros_tets_color.onnx"  
MODEL_INT8_STATIC = "/home/qualviz/QualViz/models/Classification/512_2_clstalbros_INT8_static.onnx" 
IMAGE_FOLDER      ="/home/qualviz/QualViz/calibration_images/classification/"   # update if different
OUTPUT_FOLDER     = "/home/qualviz/cls_accuracy_report/"
INPUT_NAME        = "x"
INPUT_H, INPUT_W  = 512, 512
NUM_CLASSES       = 2
CLASS_NAMES       = ["Top", "Bottom"]   # update to match your actual classes
# ------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER + "fp32/",  exist_ok=True)
os.makedirs(OUTPUT_FOLDER + "int8/",  exist_ok=True)
os.makedirs(OUTPUT_FOLDER + "diff/",  exist_ok=True)

def get_session(model_path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, opts)
    # warmup
    dummy = np.random.randn(1, 3, INPUT_H, INPUT_W).astype(np.float32)
    sess.run(None, {INPUT_NAME: dummy})
    sess.run(None, {INPUT_NAME: dummy})
    return sess

def preprocess(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (INPUT_W, INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def run_cls(session, img_input):
    """Returns predicted class index and full softmax probabilities"""
    out = session.run(None, {INPUT_NAME: img_input})[0]  # shape [1, 2]
    probs     = out[0]                          # [class0_prob, class1_prob]
    pred_cls  = int(np.argmax(probs))
    confidence= float(probs[pred_cls])
    return pred_cls, confidence, probs

def draw_cls_overlay(image, pred_cls, confidence, probs, label, ms):
    overlay = image.copy()
    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    h, w = overlay.shape[:2]
    fs = w / 1024.0
    ft = max(1, int(w / 409))

    cls_name = CLASS_NAMES[pred_cls] if pred_cls < len(CLASS_NAMES) else f"Class{pred_cls}"
    color    = (0, 255, 0) if pred_cls == 0 else (0, 0, 255)

    # Draw colored border around image
    cv2.rectangle(overlay, (5, 5), (w-5, h-5), color, 6)

    # Main prediction text
    cv2.putText(overlay,
                f"{label} | {cls_name} | {confidence*100:.1f}% | {ms:.0f}ms",
                (15, 50), cv2.FONT_HERSHEY_DUPLEX, fs, color, ft)

    # All class probabilities
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        bar_color = (0, 255, 0) if i == pred_cls else (200, 200, 200)
        bar_w = int(prob * 300)
        cv2.rectangle(overlay,
                      (15, 80 + i*40),
                      (15 + bar_w, 110 + i*40),
                      bar_color, -1)
        cv2.putText(overlay,
                    f"{name}: {prob*100:.1f}%",
                    (320, 105 + i*40),
                    cv2.FONT_HERSHEY_SIMPLEX, fs*0.8,
                    (255, 255, 255), ft)

    return overlay

def softmax_diff_stats(probs_fp32, probs_int8):
    diff = np.abs(probs_fp32 - probs_int8)
    return float(diff.mean()), float(diff.max())

# -------- Load models --------
print("Loading models...")
sess_fp32 = get_session(MODEL_FP32)
sess_int8 = get_session(MODEL_INT8_STATIC)
print("Models loaded and warmed up.\n")

# -------- Collect images --------
exts = {'.jpg', '.jpeg', '.png', '.bmp'}
image_paths = sorted([
    str(p) for p in Path(IMAGE_FOLDER).iterdir()
    if p.suffix.lower() in exts
])
print(f"Found {len(image_paths)} images in {IMAGE_FOLDER}\n")

if len(image_paths) == 0:
    print(f"ERROR: No images found! Check IMAGE_FOLDER path: {IMAGE_FOLDER}")
    exit(1)

# -------- Per-image results --------
results = []

for img_path in image_paths:
    fname = Path(img_path).name
    print(f"Processing: {fname}")

    image = cv2.imread(img_path)
    if image is None:
        print(f"  Skipping (cannot load)")
        continue

    inp = preprocess(image)

    # FP32
    t0 = time.time()
    cls_fp32, conf_fp32, probs_fp32 = run_cls(sess_fp32, inp)
    time_fp32 = (time.time() - t0) * 1000

    # INT8
    t0 = time.time()
    cls_int8, conf_int8, probs_int8 = run_cls(sess_int8, inp)
    time_int8 = (time.time() - t0) * 1000

    # Metrics
    decision_match  = (cls_fp32 == cls_int8)
    diff_mean, diff_max = softmax_diff_stats(probs_fp32, probs_int8)
    conf_drop = abs(conf_fp32 - conf_int8) * 100  # percentage points

    name_fp32 = CLASS_NAMES[cls_fp32] if cls_fp32 < len(CLASS_NAMES) else f"Class{cls_fp32}"
    name_int8 = CLASS_NAMES[cls_int8] if cls_int8 < len(CLASS_NAMES) else f"Class{cls_int8}"

    print(f"  FP32 : class={cls_fp32} ({name_fp32}) conf={conf_fp32*100:.1f}% | {time_fp32:.0f}ms")
    print(f"  INT8 : class={cls_int8} ({name_int8}) conf={conf_int8*100:.1f}% | {time_int8:.0f}ms")
    print(f"  Match={'✓' if decision_match else '✗'}  "
          f"SoftmaxDiff={diff_mean:.4f}  ConfDrop={conf_drop:.1f}pp  "
          f"Speedup={time_fp32/max(time_int8,1):.2f}x")

    # Save overlays
    ov_fp32 = draw_cls_overlay(image, cls_fp32, conf_fp32, probs_fp32, "FP32", time_fp32)
    ov_int8 = draw_cls_overlay(image, cls_int8, conf_int8, probs_int8, "INT8", time_int8)
    cv2.imwrite(OUTPUT_FOLDER + "fp32/" + fname, ov_fp32)
    cv2.imwrite(OUTPUT_FOLDER + "int8/" + fname, ov_int8)

    # Diff image: show probability difference as heatmap bar
    diff_vis = np.zeros((200, 600, 3), dtype=np.uint8)
    cv2.putText(diff_vis, f"FP32 vs INT8 — {fname}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    for i, (name, p_fp32, p_int8) in enumerate(zip(CLASS_NAMES, probs_fp32, probs_int8)):
        y = 60 + i * 60
        # FP32 bar (blue)
        cv2.rectangle(diff_vis, (10, y), (10+int(p_fp32*400), y+20), (255,100,0), -1)
        # INT8 bar (green)
        cv2.rectangle(diff_vis, (10, y+25), (10+int(p_int8*400), y+45), (0,200,100), -1)
        cv2.putText(diff_vis, f"{name}: FP32={p_fp32*100:.1f}% INT8={p_int8*100:.1f}%",
                    (10, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    match_color = (0,255,0) if decision_match else (0,0,255)
    cv2.putText(diff_vis,
                f"Decision: {'MATCH' if decision_match else 'MISMATCH'} | "
                f"FP32={name_fp32} INT8={name_int8}",
                (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, match_color, 1)
    cv2.imwrite(OUTPUT_FOLDER + "diff/" + fname, diff_vis)

    results.append({
        "image":           fname,
        "fp32_class":      cls_fp32,
        "int8_class":      cls_int8,
        "fp32_class_name": name_fp32,
        "int8_class_name": name_int8,
        "decision_match":  decision_match,
        "fp32_confidence": round(conf_fp32 * 100, 2),
        "int8_confidence": round(conf_int8 * 100, 2),
        "confidence_drop_pp": round(conf_drop, 2),
        "fp32_probs":      [round(float(p),4) for p in probs_fp32],
        "int8_probs":      [round(float(p),4) for p in probs_int8],
        "softmax_diff_mean": round(diff_mean, 6),
        "softmax_diff_max":  round(diff_max, 6),
        "fp32_ms":         round(time_fp32, 1),
        "int8_ms":         round(time_int8, 1),
        "speedup":         round(time_fp32 / max(time_int8, 1), 2),
    })

# -------- Summary --------
total      = len(results)
matches    = sum(1 for r in results if r["decision_match"])
mismatches = total - matches

avg_fp32_ms   = np.mean([r["fp32_ms"]   for r in results])
avg_int8_ms   = np.mean([r["int8_ms"]   for r in results])
avg_speedup   = np.mean([r["speedup"]   for r in results])
avg_fp32_conf = np.mean([r["fp32_confidence"] for r in results])
avg_int8_conf = np.mean([r["int8_confidence"] for r in results])
avg_conf_drop = np.mean([r["confidence_drop_pp"] for r in results])
avg_sdiff     = np.mean([r["softmax_diff_mean"] for r in results])

# Per-class accuracy breakdown
from collections import defaultdict
class_stats = defaultdict(lambda: {"total":0, "match":0})
for r in results:
    cls = r["fp32_class_name"]
    class_stats[cls]["total"] += 1
    if r["decision_match"]:
        class_stats[cls]["match"] += 1

print("\n" + "="*55)
print("      CLASSIFICATION ACCURACY DROP REPORT")
print("="*55)
print(f"  Images tested          : {total}")
print(f"  Decision matches       : {matches}/{total} ({100*matches/total:.1f}%)")
print(f"  Decision mismatches    : {mismatches}/{total} ({100*mismatches/total:.1f}%)")
print(f"  Avg FP32 confidence    : {avg_fp32_conf:.1f}%")
print(f"  Avg INT8 confidence    : {avg_int8_conf:.1f}%")
print(f"  Avg confidence drop    : {avg_conf_drop:.1f} percentage points")
print(f"  Avg softmax diff       : {avg_sdiff:.6f}")
print(f"  FP32 avg speed         : {avg_fp32_ms:.1f} ms")
print(f"  INT8 avg speed         : {avg_int8_ms:.1f} ms")
print(f"  Speedup                : {avg_speedup:.2f}x")
print(f"\n  Per-class match rate:")
for cls_name, stats in class_stats.items():
    pct = 100 * stats["match"] / max(stats["total"], 1)
    print(f"    {cls_name:12s}: {stats['match']}/{stats['total']} ({pct:.1f}%)")
print("="*55)

if mismatches > 0:
    print("\nMismatched images:")
    for r in results:
        if not r["decision_match"]:
            print(f"  {r['image']}: FP32={r['fp32_class_name']} "
                  f"({r['fp32_confidence']:.1f}%) "
                  f"vs INT8={r['int8_class_name']} "
                  f"({r['int8_confidence']:.1f}%)")

# Save JSON
report = {
    "summary": {
        "model_fp32": MODEL_FP32,
        "model_int8": MODEL_INT8_STATIC,
        "input_size": f"{INPUT_H}x{INPUT_W}",
        "total_images": total,
        "decision_match_count": matches,
        "decision_match_pct": round(100*matches/total, 1),
        "mismatch_count": mismatches,
        "avg_fp32_confidence_pct": round(float(avg_fp32_conf), 2),
        "avg_int8_confidence_pct": round(float(avg_int8_conf), 2),
        "avg_confidence_drop_pp":  round(float(avg_conf_drop), 2),
        "avg_softmax_diff":        round(float(avg_sdiff), 6),
        "fp32_avg_ms":  round(float(avg_fp32_ms), 1),
        "int8_avg_ms":  round(float(avg_int8_ms), 1),
        "speedup":      round(float(avg_speedup), 2),
        "per_class": {k: {"match_pct": round(100*v["match"]/max(v["total"],1),1),
                          "total": v["total"]}
                      for k,v in class_stats.items()},
    },
    "per_image": results
}

with open(OUTPUT_FOLDER + "report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\nReport saved : {OUTPUT_FOLDER}report.json")
print(f"Overlays     : {OUTPUT_FOLDER}fp32/  int8/  diff/")
