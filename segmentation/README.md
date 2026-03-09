# Segmentation INT8 Quantization Pipeline

This folder contains a **step-by-step pipeline to convert a Segmentation ONNX model from FP32 to INT8** using **ONNX Runtime static QDQ quantization**.

Each script represents **one stage in the quantization workflow**, allowing you to inspect, debug, and validate the model before and after quantization.

---

# Pipeline Overview

The quantization process consists of the following stages:

1. Model inspection
2. Node analysis
3. Opset upgrade
4. Shape inference preprocessing
5. Static INT8 quantization
6. Model verification
7. Accuracy validation

---

# Folder Structure

```
segmentation/
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

Replace `MODEL_PATH` with your segmentation ONNX model.

Example:

```
MODEL_PATH=/path/to/model.onnx
```

---

# Step 1 — Model Inspection

Script:

```
1_check_model.py
```

Purpose:

This script analyzes the ONNX model and prints important information such as:

* Model input name
* Input shape
* Output name
* Opset version
* Number of nodes
* Operator types used in the graph

Why this is important:

Before quantization, we must understand the **model structure and operators** to ensure compatibility.

Example usage:

```
python3 1_check_model.py --model_path MODEL_PATH
```

---

# Step 2 — Find Unsupported Nodes

Script:

```
2_find_nodes.py
```

Purpose:

Some operators in segmentation models **cannot be quantized safely**.

This script:

* Lists all operators in the model
* Detects nodes that may break INT8 quantization
* Generates a list of nodes that should be **excluded from quantization**

Why this matters:

Avoids issues such as:

* DequantizeLinear errors
* Unsupported operations
* Incorrect quantization ranges

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

Many exported models use **older opset versions (like 11)** which cause issues with quantization.

This script upgrades the model to **ONNX opset 17**.

Benefits:

* Fixes `DequantizeLinear` errors
* Improves compatibility with ONNX Runtime
* Ensures modern operator support

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

Runs **ONNX shape inference** to fill missing tensor shapes.

Why this is required:

Quantization requires **fully defined tensor shapes** for calibration and scale calculation.

This step prepares the model for quantization.

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

Converts the FP32 model into **INT8 quantized model** using:

* Static calibration
* QDQ (Quantize-Dequantize) format
* Real calibration images

This step:

* Calculates activation ranges
* Quantizes weights
* Inserts QuantizeLinear / DequantizeLinear nodes

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

This script ensures the INT8 model works correctly.

It checks:

* Model loading in ONNX Runtime
* Inference execution
* Output tensor shape
* Runtime errors

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

Compares **FP32 vs INT8 segmentation output**.

Validation includes:

* Pixel-wise output comparison
* Difference statistics
* Performance benchmarking

This ensures quantization **does not degrade segmentation quality**.

Example usage:

```
python3 7_accuracy_report.py \
--fp32_model model_preprocessed.onnx \
--int8_model model_int8.onnx \
--image_dir /path/to/test_images/
```

---

# Final Output

After completing the pipeline, you will have:

| Model      | Description                          |
| ---------- | ------------------------------------ |
| FP32 Model | Original segmentation model          |
| INT8 Model | Optimized model for faster inference |

Benefits:

* 1.5x – 2x speed improvement
* Reduced memory usage
* Efficient deployment on edge devices

---

# Recommended Workflow

Always run scripts in this order:

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

* Use **50–200 calibration images** for best results
* Ensure calibration images match training distribution
* Validate outputs before deployment
