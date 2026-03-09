import onnxruntime as ort
import numpy as np, time, os

models = [
    ("/home/qualviz/QualViz/models/anomaly/256_1_anmly_anomaly_preprocessed.onnx",   "FP32"),
    ("/home/qualviz/QualViz/models/anomaly/anomaly_int8_v2.onnx",         "INT8 v2 (QInt8)"),
    ("/home/qualviz/QualViz/models/anomaly/anomaly_int8_v3.onnx",         "INT8 v3 (QUInt8)"),
]

opts = ort.SessionOptions()
opts.intra_op_num_threads = 3
dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)

for path, label in models:
    if not os.path.exists(path):
        print(f"{label}: NOT FOUND"); continue
    sess = ort.InferenceSession(path, opts)
    sess.run(None, {"inpt.1": dummy})
    sess.run(None, {"inpt.1": dummy})
    times = []
    for _ in range(5):
        t = time.time()
        out = sess.run(None, {"inpt.1": dummy})
        times.append((time.time()-t)*1000)
    avg = sum(times)/len(times)
    print(f"{label}: {avg:.1f}ms | anomaly={bool(out[1][0])} | speedup={1808/avg:.2f}x")
