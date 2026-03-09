from onnxruntime.quantization import shape_inference
import onnx

MODEL_IN  = "/home/qualviz/QualViz/models/anomaly/256_1_anmly_anomaly_rel_path_test_color.onnx" # update your actual filename
MODEL_OUT = "/home/qualviz/QualViz/models/anomaly/256_1_anmly_anomaly_preprocessed.onnx"

print("Preprocessing...")
shape_inference.quant_pre_process(MODEL_IN, MODEL_OUT, skip_optimization=False)
print(f"Saved: {MODEL_OUT}")

# Verify
model = onnx.load(MODEL_OUT)
print(f"Opset: {model.opset_import[0].version}")
print("Done!")
