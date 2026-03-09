
import onnxruntime as ort, numpy as np, time

FP32 = "/home/qualviz/QualViz/models/Classification/256/256_2_clstalbros_preprocessed.onnx"
INT8 =  "/home/qualviz/QualViz/models/Classification/256/256_2_clstalbros_INT8_static.onnx" 

opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)

for path, label in [(FP32,"FP32"), (INT8,"INT8")]:
    sess = ort.InferenceSession(path, opts)
    sess.run(None, {"x": dummy}); sess.run(None, {"x": dummy})  # warmup
    times = []
    for _ in range(5):
        t = time.time()
        out = sess.run(None, {"x": dummy})
        times.append((time.time()-t)*1000)
    avg = sum(times)/len(times)
    pred = int(np.argmax(out[0], axis=1)[0])
    print(f"{label}: {avg:.1f}ms | class={pred} | probs={out[0]}")

