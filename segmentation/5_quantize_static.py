import onnx
import os
import numpy as np
import cv2
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from onnxruntime.quantization import quant_pre_process

# ============ CHANGE THESE ============
MODEL_PATH    = "/home/qualviz/QualViz/models/Classification/512_2_clstalbros_opset13.onnx"      # <-- your FP32 model
OUTPUT_PATH   = "/home/qualviz/QualViz/models/Classification/512_2_clstalbros_INT8_static.onnx"  # <-- output
CALIB_IMG_DIR = "/home/qualviz/QualViz/calibration_images/classification"     # <-- your images
INPUT_NAME      = "x"       # <-- from check_model.py Step 2
INPUT_H         = 512
INPUT_W         = 512
MAX_CALIB       = 50
# ======================================

# Auto-detect ALL nodes that cannot be INT8
problematic_ops = [
    "ReduceMax", "ReduceMean", "ReduceMin", "ReduceSum", "ReduceProd",
    "Resize", "Upsample", "Softmax", "Sigmoid", "LogSoftmax",
    "ArgMax", "ArgMin", "TopK", "GridSample", "NonMaxSuppression",
    "BilinearInterp", "NearestInterp"
]

model = onnx.load(MODEL_PATH)
nodes_to_exclude = [
    node.name for node in model.graph.node
    if node.op_type in problematic_ops
]
print(f"Excluding {len(nodes_to_exclude)} incompatible nodes from INT8")


class CalibReader(CalibrationDataReader):
    def __init__(self):
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        self.images = [
            os.path.join(CALIB_IMG_DIR, f)
            for f in os.listdir(CALIB_IMG_DIR)
            if f.lower().endswith(exts)
        ][:MAX_CALIB]
        self.idx = 0
        print(f"Found {len(self.images)} calibration images\n")

    def get_next(self):
        if self.idx >= len(self.images):
            return None
        img = cv2.imread(self.images[self.idx])
        self.idx += 1
        if img is None:
            return self.get_next()
        img = cv2.resize(img, (INPUT_W, INPUT_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std
        img  = np.transpose(img, (2, 0, 1))
        img  = np.expand_dims(img, axis=0)
        print(f"  Calibrating [{self.idx}/{len(self.images)}]")
        return {INPUT_NAME: img}


print("Starting static INT8 quantization...")
print(f"Model  : {MODEL_PATH}")
print(f"Output : {OUTPUT_PATH}")
print(f"Format : QDQ\n")

quantize_static(
    model_input=MODEL_PATH,
    model_output=OUTPUT_PATH,
    calibration_data_reader=CalibReader(),
    quant_format=QuantFormat.QDQ,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=True,
    nodes_to_exclude=nodes_to_exclude,
)

orig = os.path.getsize("/home/qualviz/QualViz/models/512_semsons_preprocessed.onnx") / (1024*1024)
new  = os.path.getsize(OUTPUT_PATH) / (1024*1024)
print(f"\nQuantization complete!")
print(f"Original FP32 : {orig:.1f} MB")
print(f"INT8 Static   : {new:.1f} MB")
print(f"Size reduction: {((orig-new)/orig)*100:.1f}%")
