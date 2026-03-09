from onnxruntime.quantization import quantize_static, CalibrationDataReader
from onnxruntime.quantization import QuantFormat, QuantType
import onnx, numpy as np, cv2, os

MODEL_PATH  = "/home/qualviz/QualViz/models/anomaly/256_1_anmly_anomaly_preprocessed.onnx"
OUTPUT_PATH = "/home/qualviz/QualViz/models/anomaly/anomaly_int8_v2.onnx"
CALIB_DIR   = "/home/qualviz/QualViz/calibration_images/anomaly"
INPUT_NAME  = "inpt.1"

class Reader(CalibrationDataReader):
    def __init__(self):
        exts = {'.jpg','.jpeg','.png','.bmp'}
        self.imgs = sorted([os.path.join(CALIB_DIR,f)
                           for f in os.listdir(CALIB_DIR)
                           if os.path.splitext(f)[1].lower() in exts])
        print(f"Calibration images: {len(self.imgs)}")
        self.idx = 0
    def get_next(self):
        if self.idx >= len(self.imgs): return None
        img = cv2.imread(self.imgs[self.idx]); self.idx += 1
        if img is None: return self.get_next()
        if len(img.shape)==2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (256,256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.0
        img = np.transpose(img,(2,0,1))
        return {INPUT_NAME: np.expand_dims(img,0)}

# Only exclude post-processing ops, NOT Conv layers
model = onnx.load(MODEL_PATH)
nodes_to_exclude = [n.name for n in model.graph.node
                    if n.op_type in ["Resize","ArgMax","TopK",
                                     "Softmax","Greater","Clip",
                                     "NonMaxSuppression"]]
print(f"Excluding {len(nodes_to_exclude)} nodes")

quantize_static(
    model_input=MODEL_PATH,
    model_output=OUTPUT_PATH,
    calibration_data_reader=Reader(),
    quant_format=QuantFormat.QOperator,  # QOperator instead of QDQ
    per_channel=False,                   # False for better Conv quantization
    reduce_range=True,
    weight_type=QuantType.QInt8,
    nodes_to_exclude=nodes_to_exclude,
)

int8_model = onnx.load(OUTPUT_PATH)
qlinear_conv = sum(1 for n in int8_model.graph.node if n.op_type=="QLinearConv")
print(f"\nQLinearConv ops: {qlinear_conv}")  # should be >0 now
print(f"Size: {os.path.getsize(OUTPUT_PATH)/1024/1024:.1f} MB")
print(f"Saved: {OUTPUT_PATH}")
