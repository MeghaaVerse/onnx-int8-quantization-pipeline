
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
import onnx, onnxruntime as ort, numpy as np, cv2, os

MODEL_PATH  = "/home/qualviz/QualViz/models/Classification/256/256_2_clstalbros_opset13.onnx" # or preprocessed if opset already 13
OUTPUT_PATH = "/home/qualviz/QualViz/models/Classification/256/256_2_clstalbros_INT8_static.onnx" 
CALIB_DIR   = "/home/qualviz/QualViz/calibration_images/classification/"    # update if cls images are elsewhere

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
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))
        return {"x": np.expand_dims(img, 0)}

model = onnx.load(MODEL_PATH)
nodes_to_exclude = [n.name for n in model.graph.node
                    if n.op_type in ["Softmax","ReduceMax","ReduceMean","Resize","ArgMax"]]
print(f"Excluding {len(nodes_to_exclude)} nodes: {nodes_to_exclude}")

quantize_static(
    model_input=MODEL_PATH,
    model_output=OUTPUT_PATH,
    calibration_data_reader=Reader(),
    quant_format=QuantFormat.QDQ,
    per_channel=True,
    reduce_range=True,
    weight_type=QuantType.QInt8,
    nodes_to_exclude=nodes_to_exclude,
)

orig = os.path.getsize(MODEL_PATH)/(1024*1024)
new  = os.path.getsize(OUTPUT_PATH)/(1024*1024)
print(f"\nDone! {orig:.1f}MB -> {new:.1f}MB ({((orig-new)/orig)*100:.1f}% smaller)")
print(f"Saved: {OUTPUT_PATH}")

