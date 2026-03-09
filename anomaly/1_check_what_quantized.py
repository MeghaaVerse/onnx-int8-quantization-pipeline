import onnx

FP32 = "/home/qualviz/QualViz/models/anomaly/256_1_anmly_anomaly_rel_path_test_color.onnx"
INT8 = "/home/qualviz/QualViz/models/anomaly/anomaly_256_int8_static.onnx" 

fp32_model = onnx.load(FP32)
int8_model  = onnx.load(INT8)

fp32_ops = {}
int8_ops = {}

for n in fp32_model.graph.node:
    fp32_ops[n.op_type] = fp32_ops.get(n.op_type, 0) + 1

for n in int8_model.graph.node:
    int8_ops[n.op_type] = int8_ops.get(n.op_type, 0) + 1

print(f"FP32 total nodes: {len(fp32_model.graph.node)}")
print(f"INT8 total nodes: {len(int8_model.graph.node)}")
print(f"\nINT8 quantized ops added:")
for op in ["QLinearConv","QLinearMatMul","QuantizeLinear","DequantizeLinear"]:
    print(f"  {op}: {int8_ops.get(op, 0)}")
print(f"\nOps still in FP32:")
for op, count in sorted(fp32_ops.items(), key=lambda x: -x[1])[:10]:
    print(f"  {op}: {count}")
