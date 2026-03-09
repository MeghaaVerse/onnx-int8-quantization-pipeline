# Classification INT8 Quantization Pipeline

This folder contains a **step-by-step pipeline to convert an ONNX Classification model from FP32 to INT8** using **ONNX Runtime static QDQ quantization**.

The scripts allow you to **inspect, prepare, quantize, and validate classification models** while ensuring minimal accuracy loss.

Each script represents **one stage of the quantization process**.

---

# Pipeline Overview

The classification quantization workflow consists of the following stages:

1. Model inspection
2. Node analysis
3. Opset upgrade
4. Shape inference preprocessing
5. Static INT8 quantization
6. Model verification
7. Accuracy comparison

---

# Folder Structure

```
classification/
│
├── 1_check_model.py
├── 2_find_nodes.py
├── 3_upgrade_opset.py
├── 4_preprocess_model.py
├── 5_quantize_static.py
├── 6_verify_model.py
└── 7_accuracy_report.py
```

---

# Step-by-Step Usage

Replace `MODEL_PATH` with your classification ONNX model.

Example:

```
MODEL_PATH=/path/to/classification_model.onnx
```

---

# Step 1 — Model Inspection

Script:

```
1_check_model.py
```

Purpose:

This script inspects the ONNX classification model and prints important information such as:

* Model input name
* Input tensor shape
* Output tensor name
* Opset version
* Number of nodes
* Operator types used

Why this step matters:

Understanding the **model architecture and operators** helps ensure the model is compatible with INT8 quantization.

Example usage:

```
python3 1_check_model.py --model_path MODEL_PATH
```

---

# Step 2 — Find Quantization-Sensitive Nodes

Script:

```
2_find_nodes.py
```

Purpose:

Some operators may **not be suitable for quantization**.

This script:

* Lists all operators in the model
* Identifies nodes that may cause quantization issues
* Helps determine nodes that should be excluded

Why this matters:

Prevents problems like:

* Unsupported quantized operators
* Incorrect activation ranges
* Runtime failures after quantization

Example usage:

```
python3 2_find_nodes.py --model_path MODEL_PATH
```

---

# Step 3 — Upgrade ONNX Opset

Script:

```
3_upgrade_opset.py
```

Purpose:

Many classification models are exported with **older ONNX opsets (for example opset 11)**.

This script upgrades the model to **opset 17**.

Benefits:

* Fixes `QuantizeLinear` / `DequantizeLinear` compatibility issues
* Improves ONNX Runtime support
* Enables modern operator behavior

Example usage:

```
python3 3_upgrade_opset.py --model_path MODEL_PATH
```

Output:

```
model_opset17.onnx
```

---

# Step 4 — Shape Inference Preprocessing

Script:

```
4_preprocess_model.py
```

Purpose:

This step runs **ONNX shape inference**.

It automatically fills missing tensor shapes in the graph.

Why this step is required:

Static quantization requires **fully defined tensor shapes** to calculate calibration ranges.

Example usage:

```
python3 4_preprocess_model.py --model_path model_opset17.onnx
```

Output:

```
model_preprocessed.onnx
```

---

# Step 5 — Static INT8 Quantization

Script:

```
5_quantize_static.py
```

Purpose:

This script performs **static INT8 quantization** using calibration images.

It:

* Calculates activation ranges
* Quantizes weights
* Inserts QDQ nodes into the model graph

Quantization format used:

**QDQ (Quantize-Dequantize)**

Example usage:

```
python3 5_quantize_static.py \
--model_path model_preprocessed.onnx \
--calib_dir /path/to/calibration_images/
```

Output:

```
model_int8.onnx
```

---

# Step 6 — Verify Quantized Model

Script:

```
6_verify_model.py
```

Purpose:

This script verifies that the quantized model runs correctly.

Checks performed:

* Model loading in ONNX Runtime
* Inference execution
* Output tensor shape validation
* Runtime error detection

Example usage:

```
python3 6_verify_model.py --model_path model_int8.onnx
```

---

# Step 7 — Accuracy Comparison

Script:

```
7_accuracy_report.py
```

Purpose:

Compares **FP32 vs INT8 classification outputs**.

Validation includes:

* Top-1 class prediction comparison
* Probability distribution similarity
* Performance benchmarking

This ensures quantization **does not significantly impact classification accuracy**.

Example usage:

```
python3 7_accuracy_report.py \
--fp32_model model_preprocessed.onnx \
--int8_model model_int8.onnx \
--image_dir /path/to/test_images/
```

---

# Final Output

After completing the pipeline, you will obtain:

| Model      | Description                          |
| ---------- | ------------------------------------ |
| FP32 Model | Original classification model        |
| INT8 Model | Optimized model for faster inference |

Benefits:

* Faster inference (1.5x – 2x speedup)
* Reduced model size
* Efficient edge deployment

---

# Recommended Execution Order

Always run the scripts in the following order:

```
1_check_model.py
2_find_nodes.py
3_upgrade_opset.py
4_preprocess_model.py
5_quantize_static.py
6_verify_model.py
7_accuracy_report.py
```

---

# Notes

* Use **50–200 calibration images** for accurate quantization
* Calibration images should represent the real data distribution
* Always verify accuracy before deployment
