import onnx
from onnx import version_converter

MODEL_PATH   = "/home/qualviz/QualViz/models/Classification/512_2_clstalbros_preprocessed.onnx"
OUTPUT_PATH  = "/home/qualviz/QualViz/models/512_2_clstalbros_opset13.onnx"

print("Loading model...")
model = onnx.load(MODEL_PATH)
print(f"Current opset: {model.opset_import[0].version}")

print("Converting to opset 13...")
converted = version_converter.convert_version(model, 13)

onnx.save(converted, OUTPUT_PATH)
print(f"Saved to: {OUTPUT_PATH}")

# Verify
m2 = onnx.load(OUTPUT_PATH)
print(f"New opset: {m2.opset_import[0].version}")

# Validate
onnx.checker.check_model(m2)
print("Model validation passed!")
