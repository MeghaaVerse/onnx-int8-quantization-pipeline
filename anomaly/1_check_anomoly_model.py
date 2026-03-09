
import onnx
import onnxruntime as ort

# UPDATE this path
MODEL = "/home/qualviz/QualViz/models/anomaly/256_1_anmly_anomaly_rel_path_test_color.onnx"

model = onnx.load(MODEL)
print(f"Opset: {model.opset_import[0].version}")

sess = ort.InferenceSession(MODEL)
for i in sess.get_inputs():
    print(f"Input : {i.name} {i.shape} {i.type}")
for o in sess.get_outputs():
    print(f"Output: {o.name} {o.shape} {o.type}")

problematic_ops = ["Softmax","ReduceMax","ReduceMean","Resize",
                   "Sigmoid","ArgMax","GlobalAveragePool","TopK"]
nodes_to_exclude = [n.name for n in model.graph.node
                    if n.op_type in problematic_ops]
print(f"\nNodes to exclude: {len(nodes_to_exclude)}")
for n in nodes_to_exclude:
    print(f"  {n}")
