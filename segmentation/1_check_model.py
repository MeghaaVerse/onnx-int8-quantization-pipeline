import onnxruntime as ort
import onnx

model_path = "/home/qualviz/QualViz/models/Classification/256/256_2_clstalbros_meta_data_test_2_color.onnx";  

# Check inputs and outputs
sess = ort.InferenceSession(model_path)

print("=== INPUTS ===")
for inp in sess.get_inputs():
    print(f"  Name : {inp.name}")
    print(f"  Shape: {inp.shape}")
    print(f"  Type : {inp.type}")

print("\n=== OUTPUTS ===")
for out in sess.get_outputs():
    print(f"  Name : {out.name}")
    print(f"  Shape: {out.shape}")
    print(f"  Type : {out.type}")
