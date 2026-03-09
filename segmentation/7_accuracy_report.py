import onnxruntime as ort
import numpy as np
import cv2
import os
import time
import json
from pathlib import Path

# -------- CONFIG --------
MODEL_FP32        = "/home/qualviz/QualViz/models/512_2_semsons_mode_512_3_channel_color.onnx"
MODEL_INT8_STATIC = "/home/qualviz/QualViz/models/512_semsons_onnx_static_int8.onnx"
IMAGE_FOLDER      = "/home/qualviz/QualViz/calibration_images"
OUTPUT_FOLDER     = "/home/qualviz/accuracy_report/"
INPUT_NAME        = "x"
INPUT_H, INPUT_W  = 512, 512
MIN_AREA          = 49
CONF_THRESHOLD    = 63
# ------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER + "fp32/",  exist_ok=True)
os.makedirs(OUTPUT_FOLDER + "int8/",  exist_ok=True)
os.makedirs(OUTPUT_FOLDER + "diff/",  exist_ok=True)

def get_session(model_path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, opts)

def preprocess(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (INPUT_W, INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def get_defect_mask(session, img_input):
    out = session.run(None, {INPUT_NAME: img_input})[0]  # [1,2,512,512]
    class0 = out[0, 0]
    class1 = out[0, 1]
    defect_mask = (class1 > class0).astype(np.uint8) * 255
    conf_mask   = (class1 * 100).astype(np.uint8)
    return defect_mask, conf_mask, out

def get_valid_contours(defect_mask, conf_mask, orig_w, orig_h):
    mask_resized = cv2.resize(defect_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    conf_resized = cv2.resize(conf_mask,   (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask_resized, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        rr   = cv2.minAreaRect(c)
        w, h = rr[1]
        if w == 0 or h == 0: continue
        if h > 8 * w: valid.append(c); continue
        if area < MIN_AREA: continue

        msk_tmp = np.zeros(mask_resized.shape, dtype=np.uint8)
        cv2.drawContours(msk_tmp, [c], -1, 255, -1)
        total_px  = cv2.countNonZero(msk_tmp)
        nz_px     = cv2.countNonZero(cv2.bitwise_and(conf_resized, msk_tmp))

        if nz_px + 1000 < total_px:
            valid.append(c); continue

        mean_conf = cv2.mean(conf_resized, mask=msk_tmp)[0]
        if mean_conf >= CONF_THRESHOLD:
            valid.append(c)

    return valid, mask_resized, conf_resized

def draw_overlay(image, contours, label, ms):
    overlay = image.copy()
    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    fs = overlay.shape[1] / 1024.0
    ft = max(1, int(overlay.shape[1] / 409))
    result = "FAIL" if contours else "PASS"
    color  = (0, 0, 255) if contours else (0, 255, 0)
    cv2.drawContours(overlay, contours, -1, (0,0,255), 2)
    cv2.putText(overlay, f"{label} | {result} | {ms:.0f}ms | defects:{len(contours)}",
                (10, 40), cv2.FONT_HERSHEY_DUPLEX, fs, color, ft)
    return overlay, result

def pixel_iou(mask_a, mask_b):
    """IoU between two binary masks"""
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or( mask_a > 0, mask_b > 0).sum()
    return float(inter) / float(union) if union > 0 else 1.0

def mask_dice(mask_a, mask_b):
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    denom = (mask_a > 0).sum() + (mask_b > 0).sum()
    return 2.0 * inter / denom if denom > 0 else 1.0

# -------- Load models --------
print("Loading models...")
sess_fp32 = get_session(MODEL_FP32)
sess_int8 = get_session(MODEL_INT8_STATIC)
print("Models loaded.\n")

# -------- Collect images --------
exts = {'.jpg', '.jpeg', '.png', '.bmp'}
image_paths = sorted([
    str(p) for p in Path(IMAGE_FOLDER).iterdir()
    if p.suffix.lower() in exts
])
print(f"Found {len(image_paths)} images\n")

# -------- Per-image results --------
results = []

for img_path in image_paths:
    fname = Path(img_path).name
    print(f"Processing: {fname}")

    image = cv2.imread(img_path)
    if image is None:
        print(f"  Skipping (cannot load)")
        continue

    orig_h, orig_w = image.shape[:2]
    inp = preprocess(image)

    # FP32
    t0 = time.time()
    mask_fp32, conf_fp32, raw_fp32 = get_defect_mask(sess_fp32, inp)
    time_fp32 = (time.time() - t0) * 1000
    contours_fp32, mask_full_fp32, _ = get_valid_contours(mask_fp32, conf_fp32, orig_w, orig_h)

    # INT8
    t0 = time.time()
    mask_int8, conf_int8, raw_int8 = get_defect_mask(sess_int8, inp)
    time_int8 = (time.time() - t0) * 1000
    contours_int8, mask_full_int8, _ = get_valid_contours(mask_int8, conf_int8, orig_w, orig_h)

    # Metrics
    iou   = pixel_iou(mask_full_fp32, mask_full_int8)
    dice  = mask_dice(mask_full_fp32, mask_full_int8)
    px_fp32 = int(cv2.countNonZero(mask_full_fp32))
    px_int8 = int(cv2.countNonZero(mask_full_int8))
    px_diff_pct = abs(px_fp32 - px_int8) / max(px_fp32, 1) * 100

    # Softmax output difference
    diff_mean = float(np.abs(raw_fp32 - raw_int8).mean())
    diff_max  = float(np.abs(raw_fp32 - raw_int8).max())

    result_fp32 = "FAIL" if contours_fp32 else "PASS"
    result_int8 = "FAIL" if contours_int8 else "PASS"
    decision_match = (result_fp32 == result_int8)

    print(f"  FP32 : {result_fp32} | defects={len(contours_fp32)} | px={px_fp32} | {time_fp32:.0f}ms")
    print(f"  INT8 : {result_int8} | defects={len(contours_int8)} | px={px_int8} | {time_int8:.0f}ms")
    print(f"  IoU={iou:.4f}  Dice={dice:.4f}  PixDiff={px_diff_pct:.1f}%  Match={'✓' if decision_match else '✗'}")

    # Save overlays
    ov_fp32, _ = draw_overlay(image, contours_fp32, "FP32",  time_fp32)
    ov_int8, _ = draw_overlay(image, contours_int8, "INT8",  time_int8)

    cv2.imwrite(OUTPUT_FOLDER + "fp32/" + fname, ov_fp32)
    cv2.imwrite(OUTPUT_FOLDER + "int8/" + fname, ov_int8)

    # Diff image: red=FP32 only, blue=INT8 only, green=both agree
    diff_vis = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    fp32_bin = mask_full_fp32 > 0
    int8_bin = mask_full_int8 > 0
    diff_vis[fp32_bin & ~int8_bin]  = (0, 0, 255)   # FP32 only  → red
    diff_vis[int8_bin & ~fp32_bin]  = (255, 0, 0)   # INT8 only  → blue
    diff_vis[fp32_bin &  int8_bin]  = (0, 255, 0)   # both agree → green
    cv2.putText(diff_vis, "RED=FP32only  BLUE=INT8only  GREEN=both",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.imwrite(OUTPUT_FOLDER + "diff/" + fname, diff_vis)

    results.append({
        "image":          fname,
        "fp32_result":    result_fp32,
        "int8_result":    result_int8,
        "decision_match": decision_match,
        "fp32_defects":   len(contours_fp32),
        "int8_defects":   len(contours_int8),
        "fp32_pixels":    px_fp32,
        "int8_pixels":    px_int8,
        "pixel_diff_pct": round(px_diff_pct, 2),
        "iou":            round(iou, 4),
        "dice":           round(dice, 4),
        "softmax_diff_mean": round(diff_mean, 6),
        "softmax_diff_max":  round(diff_max, 6),
        "fp32_ms":        round(time_fp32, 1),
        "int8_ms":        round(time_int8, 1),
        "speedup":        round(time_fp32 / max(time_int8, 1), 2),
    })

# -------- Summary --------
total       = len(results)
matches     = sum(1 for r in results if r["decision_match"])
mismatches  = total - matches
avg_iou     = np.mean([r["iou"]  for r in results])
avg_dice    = np.mean([r["dice"] for r in results])
avg_pxdiff  = np.mean([r["pixel_diff_pct"] for r in results])
avg_fp32_ms = np.mean([r["fp32_ms"] for r in results])
avg_int8_ms = np.mean([r["int8_ms"] for r in results])
avg_speedup = np.mean([r["speedup"] for r in results])
avg_sdiff   = np.mean([r["softmax_diff_mean"] for r in results])

print("\n" + "="*55)
print("           ACCURACY DROP REPORT")
print("="*55)
print(f"  Images tested         : {total}")
print(f"  Decision matches      : {matches}/{total} ({100*matches/total:.1f}%)")
print(f"  Decision mismatches   : {mismatches}/{total} ({100*mismatches/total:.1f}%)")
print(f"  Avg IoU (mask)        : {avg_iou:.4f}  (1.0 = perfect)")
print(f"  Avg Dice (mask)       : {avg_dice:.4f}  (1.0 = perfect)")
print(f"  Avg pixel diff        : {avg_pxdiff:.1f}%")
print(f"  Avg softmax diff      : {avg_sdiff:.6f}")
print(f"  FP32 avg speed        : {avg_fp32_ms:.1f} ms")
print(f"  INT8 avg speed        : {avg_int8_ms:.1f} ms")
print(f"  Speedup               : {avg_speedup:.2f}x")
print("="*55)

if mismatches > 0:
    print("\nMismatched images:")
    for r in results:
        if not r["decision_match"]:
            print(f"  {r['image']}: FP32={r['fp32_result']} INT8={r['int8_result']}")

# Save JSON report
report = {
    "summary": {
        "total_images": total,
        "decision_match_count": matches,
        "decision_match_pct": round(100*matches/total, 1),
        "mismatch_count": mismatches,
        "avg_iou": round(float(avg_iou), 4),
        "avg_dice": round(float(avg_dice), 4),
        "avg_pixel_diff_pct": round(float(avg_pxdiff), 2),
        "avg_softmax_diff": round(float(avg_sdiff), 6),
        "fp32_avg_ms": round(float(avg_fp32_ms), 1),
        "int8_avg_ms": round(float(avg_int8_ms), 1),
        "speedup": round(float(avg_speedup), 2),
    },
    "per_image": results
}

with open(OUTPUT_FOLDER + "report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\nFull report saved to: {OUTPUT_FOLDER}report.json")
print(f"Overlays saved to   : {OUTPUT_FOLDER}fp32/  int8/  diff/")
