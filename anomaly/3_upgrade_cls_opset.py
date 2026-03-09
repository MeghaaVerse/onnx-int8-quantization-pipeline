import onnx
from onnx.version_converter import convert_version

MODEL_IN  = "/home/qualviz/QualViz/models/Classification/256/256_2_clstalbros_preprocessed.onnx"
MODEL_OUT = "/home/qualviz/QualViz/models/Classification/256/256_2_clstalbros_opset13.onnx"

model = onnx.load(MODEL_IN)
print(f"Original opset: {model.opset_import[0].version}")

converted = convert_version(model, 13)
onnx.save(converted, MODEL_OUT)
print(f"Upgraded opset: {converted.opset_import[0].version}")
print(f"Saved: {MODEL_OUT}")
