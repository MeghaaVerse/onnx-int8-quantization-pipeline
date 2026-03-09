import onnxruntime as ort
import numpy as np, cv2, os, time
from pathlib import Path

FP32 = "/home/qualviz/QualViz/models/anomaly/256_1_anmly_anomaly_preprocessed.onnx"
INT8 = "/home/qualviz/QualViz/models/anomaly/anomaly_256_int8_static.onnx"
TEST_DIR  ="/home/qualviz/QualViz/calibration_images/anomaly/"    # update if different
REPORT_DIR = "/home/qualviz/anomaly_accuracy_report"
INPUT_NAME = "inpt.1"
INPUT_H, INPUT_W = 256, 256

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(f"{REPORT_DIR}/overlays", exist_ok=True)

opts = ort.SessionOptions()
opts.intra_op_num_threads = 3
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

fp32_sess = ort.InferenceSession(FP32, opts)
int8_sess  = ort.InferenceSession(INT8, opts)

exts = {'.jpg','.jpeg','.png','.bmp'}
imgs = sorted([f for f in Path(TEST_DIR).iterdir()
               if f.suffix.lower() in exts])
print(f"Test images: {len(imgs)}")

results = []
fp32_times, int8_times = [], []

for img_path in imgs:
    img = cv2.imread(str(img_path))
    if img is None: continue
    orig = img.copy()

    img_resized = cv2.resize(img, (INPUT_W, INPUT_H))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    blob = (img_rgb.astype(np.float32) / 255.0)
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, 0)

    # FP32 inference
    t = time.time()
    fp32_out = fp32_sess.run(None, {INPUT_NAME: blob})
    fp32_times.append((time.time()-t)*1000)

    fp32_score   = float(fp32_out[0][0])
    fp32_anomaly = bool(fp32_out[1][0])
    fp32_map     = fp32_out[2][0][0]  # [224,224]

    # INT8 inference
    t = time.time()
    int8_out = int8_sess.run(None, {INPUT_NAME: blob})
    int8_times.append((time.time()-t)*1000)

    int8_score   = float(int8_out[0][0])
    int8_anomaly = bool(int8_out[1][0])
    int8_map     = int8_out[2][0][0]  # [224,224]

    # Decision match
    decision_match = (fp32_anomaly == int8_anomaly)
    score_diff = abs(fp32_score - int8_score)

    # Map comparison
    map_diff = np.mean(np.abs(fp32_map - int8_map))
    map_corr = np.corrcoef(fp32_map.flatten(), int8_map.flatten())[0,1]

    results.append({
        "file": img_path.name,
        "fp32_score": fp32_score,
        "int8_score": int8_score,
        "score_diff": score_diff,
        "fp32_anomaly": fp32_anomaly,
        "int8_anomaly": int8_anomaly,
        "decision_match": decision_match,
        "map_diff": map_diff,
        "map_corr": map_corr,
    })

    # ── Overlay image ──────────────────────────────────────────────────
    # Resize maps to original image size for overlay
    h, w = orig.shape[:2]
    fp32_heatmap = cv2.resize(fp32_map, (w, h))
    int8_heatmap = cv2.resize(int8_map, (w, h))
    diff_map     = np.abs(fp32_map - int8_map)
    diff_heatmap = cv2.resize(diff_map, (w, h))

    def to_colormap(m):
        m_norm = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(m_norm, cv2.COLORMAP_JET)

    fp32_color = to_colormap(fp32_heatmap)
    int8_color = to_colormap(int8_heatmap)
    diff_color = to_colormap(diff_heatmap)

    orig_bgr = cv2.resize(orig, (w, h))

    # Overlay heatmaps on original image
    fp32_overlay = cv2.addWeighted(orig_bgr, 0.6, fp32_color, 0.4, 0)
    int8_overlay = cv2.addWeighted(orig_bgr, 0.6, int8_color, 0.4, 0)

    # Labels
    fp32_label = f"FP32 | score={fp32_score:.3f} | {'ANOMALY' if fp32_anomaly else 'OK'}"
    int8_label = f"INT8 | score={int8_score:.3f} | {'ANOMALY' if int8_anomaly else 'OK'}"
    match_label = f"Decision: {'MATCH' if decision_match else 'MISMATCH'} | score_diff={score_diff:.4f} | map_corr={map_corr:.4f}"

    border_color = (0, 200, 0) if decision_match else (0, 0, 255)

    for overlay in [fp32_overlay, int8_overlay, diff_color]:
        cv2.rectangle(overlay, (0,0), (overlay.shape[1]-1, overlay.shape[0]-1), border_color, 6)

    font = cv2.FONT_HERSHEY_SIMPLEX
    def put_label(img, text, y, color=(255,255,255)):
        cv2.putText(img, text, (10, y), font, 0.6, (0,0,0), 3)
        cv2.putText(img, text, (10, y), font, 0.6, color,   1)

    put_label(fp32_overlay, fp32_label, 30)
    put_label(int8_overlay, int8_label, 30)
    put_label(diff_color,   f"Heatmap diff (mean={map_diff:.4f})", 30)

    # Resize panels to same height
    panel_h = 300
    panel_w = int(w * panel_h / h)
    p1 = cv2.resize(fp32_overlay, (panel_w, panel_h))
    p2 = cv2.resize(int8_overlay, (panel_w, panel_h))
    p3 = cv2.resize(diff_color,   (panel_w, panel_h))

    # Bottom bar with match info
    bar = np.zeros((40, panel_w*3, 3), dtype=np.uint8)
    bar_color = (0,180,0) if decision_match else (0,0,220)
    bar[:] = bar_color
    cv2.putText(bar, match_label, (10, 28), font, 0.55, (255,255,255), 1)

    composite = np.hstack([p1, p2, p3])
    composite = np.vstack([composite, bar])

    out_path = f"{REPORT_DIR}/overlays/{img_path.stem}_compare.jpg"
    cv2.imwrite(out_path, composite)

    status = "MATCH" if decision_match else "MISMATCH"
    print(f"[{status}] {img_path.name:30s} | FP32={fp32_score:.3f}({'A' if fp32_anomaly else 'O'}) "
          f"INT8={int8_score:.3f}({'A' if int8_anomaly else 'O'}) "
          f"score_diff={score_diff:.4f} map_corr={map_corr:.3f}")

# ── Summary ────────────────────────────────────────────────────────────────
total = len(results)
matches = sum(1 for r in results if r["decision_match"])
mismatches = total - matches
avg_score_diff = np.mean([r["score_diff"] for r in results])
avg_map_diff   = np.mean([r["map_diff"]   for r in results])
avg_map_corr   = np.mean([r["map_corr"]   for r in results])
avg_fp32_time  = np.mean(fp32_times)
avg_int8_time  = np.mean(int8_times)

print(f"\n{'='*65}")
print(f"ANOMALY ACCURACY REPORT  |  FP32 vs INT8 ")
print(f"{'='*65}")
print(f"Total images       : {total}")
print(f"Decision match     : {matches}/{total} ({100*matches/total:.1f}%)")
print(f"Decision mismatch  : {mismatches}")
print(f"Avg score diff     : {avg_score_diff:.4f}")
print(f"Avg heatmap diff   : {avg_map_diff:.4f}")
print(f"Avg map correlation: {avg_map_corr:.4f}  (1.0 = perfect)")
print(f"Avg FP32 time      : {avg_fp32_time:.1f} ms")
print(f"Avg INT8 time      : {avg_int8_time:.1f} ms")
print(f"Speedup            : {avg_fp32_time/avg_int8_time:.2f}x")
print(f"Report saved to    : {REPORT_DIR}/overlays/")

if mismatches > 0:
    print(f"\nMismatched images:")
    for r in results:
        if not r["decision_match"]:
            print(f"  {r['file']} | FP32={'ANOMALY' if r['fp32_anomaly'] else 'OK'} "
                  f"INT8={'ANOMALY' if r['int8_anomaly'] else 'OK'} "
                  f"score_diff={r['score_diff']:.4f}")
