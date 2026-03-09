import onnxruntime as ort
import numpy as np, time, os

# Update these paths
FP32 = "/home/qualviz/QualViz/models/anomaly/256_1_anmly_anomaly_rel_path_test_color.onnx"
INT8 = "/home/qualviz/QualViz/models/anomaly/anomaly_256_int8_static.onnx"
# update if different

print(f"FP32 size: {os.path.getsize(FP32)/1024/1024:.1f} MB")
print(f"INT8 size: {os.path.getsize(INT8)/1024/1024:.1f} MB")

opts = ort.SessionOptions()
opts.intra_op_num_threads = 3
dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)

for path, label in [(FP32,"FP32"), (INT8,"INT8")]:
    sess = ort.InferenceSession(path, opts)
    sess.run(None, {"inpt.1": dummy})  # warmup
    t = time.time()
    for _ in range(3):
        out = sess.run(None, {"inpt.1": dummy})
    avg = (time.time()-t)/3*1000
    print(f"{label}: {avg:.1f}ms | score={float(out[0][0]):.3f} | anomaly={bool(out[1][0])}")

