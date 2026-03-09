from onnxruntime.quantization import quant_pre_process
import onnx

MODEL_PATH     = "/home/qualviz/QualViz/models/Classification/512_2_clstalbros_tets_color.onnx" 
PROCESSED_PATH = "/home/qualviz/QualViz/models/Classification/512_2_clstalbros_preprocessed.onnx";

print("Pre-processing model for quantization...")

quant_pre_process(
    input_model_path=MODEL_PATH,
    output_model_path=PROCESSED_PATH,
    skip_optimization=False,
    skip_onnx_shape=False,
    skip_symbolic_shape=False,
    auto_merge=True,
    int_max=2**31 - 1,
    verbose=1
)

orig = onnx.load(MODEL_PATH)
proc = onnx.load(PROCESSED_PATH)
print(f"Original nodes : {len(orig.graph.node)}")
print(f"Processed nodes: {len(proc.graph.node)}")
print(f"Saved to: {PROCESSED_PATH}")
