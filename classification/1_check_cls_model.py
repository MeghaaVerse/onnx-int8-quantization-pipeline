import onnx
import onnxruntime as ort

MODEL =  "/home/qualviz/QualViz/models/Classification/256/256_2_clstalbros_meta_data_test_2_color.onnx"

model = onnx.load(MODEL)
opset = model.opset_import[0].version
print(f"Opset: {opset}")

# Check input/output
sess = ort.InferenceSession(MODEL)
for i in sess.get_inputs():
    print(f"Input : {i.name} {i.shape} {i.type}")
for o in sess.get_outputs():
    print(f"Output: {o.name} {o.shape} {o.type}")

# Check ops that may be incompatible
problematic_ops = ["ReduceMax","ReduceMean","Resize","Sigmoid","Softmax","ArgMax"]
nodes_to_exclude = [n.name for n in model.graph.node if n.op_type in problematic_ops]
print(f"\nNodes to exclude from quantization: {len(nodes_to_exclude)}")
for n in nodes_to_exclude:
    print(f"  {n}")
