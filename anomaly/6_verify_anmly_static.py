import onnxruntime as ort
import numpy as np, time

FP32 = "/home/qualviz/QualViz/models/anomaly/256_1_anmly_anomaly_preprocessed.onnx"
INT8 =  "/home/qualviz/QualViz/models/anomaly/anomaly_256_int8_static.onnx"
INPUT_NAME = "inpt.1"

opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)

for path, label in [(FP32,"FP32"), (INT8,"INT8 Static")]:
    sess = ort.InferenceSession(path, opts)
    sess.run(None, {INPUT_NAME: dummy})  # warmup
    sess.run(None, {INPUT_NAME: dummy})
    times = []
    for _ in range(5):
        t = time.time()
        out = sess.run(None, {INPUT_NAME: dummy})
        times.append((time.time()-t)*1000)

    pred_score  = float(out[0][0])
    is_anomaly  = bool(out[1][0])
    anomaly_map = out[2]   # [1,1,224,224]
    anomaly_bin = out[3]   # [1,1,224,224] bool

    print(f"\n{label}:")
    print(f"  Avg time     : {sum(times)/len(times):.1f} ms")
    print(f"  Pred score   : {pred_score:.4f}")
    print(f"  Is anomaly   : {is_anomaly}")
    print(f"  Anomaly map  : shape={anomaly_map.shape} "
          f"min={anomaly_map.min():.3f} max={anomaly_map.max():.3f}")
    print(f"  Anomaly mask : shape={anomaly_bin.shape} "
          f"positive_pixels={anomaly_bin.sum()}")
