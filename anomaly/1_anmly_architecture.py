import onnx
MODEL = "/home/qualviz/QualViz/models/anomaly/256_1_anmly_anomaly_rel_path_test_color.onnx"
model = onnx.load(MODEL)

# Check model metadata
print("Model metadata:")
for prop in model.metadata_props:
    print(f"  {prop.key}: {prop.value}")

# Check first 5 and last 5 node names to identify architecture
nodes = list(model.graph.node)
print(f"\nTotal nodes: {len(nodes)}")
print("First 5 nodes:")
for n in nodes[:5]:
    print(f"  {n.op_type}: {n.name}")
print("Last 5 nodes:")
for n in nodes[-5:]:
    print(f"  {n.op_type}: {n.name}")
