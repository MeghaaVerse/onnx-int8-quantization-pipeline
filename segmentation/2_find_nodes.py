import onnx

MODEL_PATH = "/home/qualviz/QualViz/models/512_2_semsons_mode_512_3_channel_color_int8.onnx" ; # <-- CHANGE to your actual path

model = onnx.load(MODEL_PATH)

# Op types that are known to be incompatible with INT8 on ARM ORT
problematic_ops = [
    "ReduceMax", "ReduceMean", "ReduceMin", "ReduceSum", "ReduceProd",
    "Resize", "Upsample", "Softmax", "Sigmoid", "LogSoftmax",
    "ArgMax", "ArgMin", "TopK", "NonMaxSuppression",
    "BilinearInterp", "InterpolateLinear", "GridSample"
]

print("=== All nodes in model ===")
found_nodes = []
for node in model.graph.node:
    if node.op_type in problematic_ops:
        print(f"  [PROBLEMATIC] op={node.op_type}  name={node.name}")
        found_nodes.append(node.name)
    
print(f"\n=== Found {len(found_nodes)} problematic nodes ===")
for n in found_nodes:
    print(f'  "{n}",')

print("\nCopy the list above into nodes_to_exclude in your quantize script")
