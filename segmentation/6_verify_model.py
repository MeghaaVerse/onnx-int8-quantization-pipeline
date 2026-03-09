import onnxruntime as ort
import numpy as np
import time

MODEL_FP32 = "/home/qualviz/QualViz/models/Classification/512_2_clstalbros_preprocessed.onnx"
MODEL_INT8_STATIC  = "/home/qualviz/QualViz/models/Classificatio n/512_2_clstalbros_INT8_static.onnx" 
MODEL_INT8_DYNAMIC = "/home/qualviz/QualViz/models/512_semsons_int8_dynamic.onnx"

opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

INPUT_NAME = "x"      # from your earlier output
INPUT_H    = 512
INPUT_W    = 512

dummy = np.random.randn(1, 3, INPUT_H, INPUT_W).astype(np.float32)

def benchmark(model_path, label):
    try:
        sess = ort.InferenceSession(model_path, opts)
        # warmup
        sess.run(None, {INPUT_NAME: dummy})
        sess.run(None, {INPUT_NAME: dummy})
        # timed
        times = []
        for _ in range(5):
            t = time.time()
            out = sess.run(None, {INPUT_NAME: dummy})
            times.append((time.time() - t) * 1000)
        avg = sum(times) / len(times)

        # check output sanity
        arr = out[0]
        print(f"{label}:")
        print(f"  Avg     : {avg:.1f} ms")
        print(f"  Shape   : {arr.shape}")
        print(f"  Class0  : min={arr[0,0].min():.4f} max={arr[0,0].max():.4f}")
        print(f"  Class1  : min={arr[0,1].min():.4f} max={arr[0,1].max():.4f}")
        defect_px = int((arr[0,1] > arr[0,0]).sum())
        print(f"  Defect px (argmax=1): {defect_px} / {512*512}")
        print()
    except Exception as e:
        print(f"{label}: FAILED — {e}\n")

benchmark(MODEL_FP32,         "FP32")
benchmark(MODEL_INT8_STATIC,  "INT8 Static")
benchmark(MODEL_INT8_DYNAMIC, "INT8 Dynamic")
