```markdown
# Anomaly Detection INT8 Quantization Pipeline

This folder contains scripts to **analyze, quantize, and validate ONNX anomaly detection models** such as **PatchCore and PaDiM** using **ONNX Runtime INT8 quantization**.

Unlike classification or segmentation, anomaly models rely heavily on **feature embeddings and distance calculations**, so quantization must be performed carefully to avoid degrading anomaly detection accuracy.

This folder includes **multiple quantization experiments**:

- QDQ static quantization
- QINT8 quantization
- QUINT8 quantization

---

# Pipeline Overview

The anomaly quantization workflow includes the following stages:

1. Understanding anomaly model architecture
2. Inspecting the ONNX model
3. Checking which layers are quantized
4. Detecting unsupported nodes
5. Upgrading ONNX opset
6. Model preprocessing with shape inference
7. INT8 quantization experiments
8. Model verification and testing
9. Benchmark comparison
10. Accuracy validation

---

# Folder Structure

```

anomaly/
│
├── 1_anmly_architecture.py
├── 1_check_anomoly_model.py
├── 1_check_what_quantized.py
│
├── 2_find_nodes.py
│
├── 3_upgrade_cls_opset.py
│
├── 4_preprocess_cls_model.py
│
├── 5_QDQ_quantize_256_anmly_static.py
├── 5_Qint8_quantize_anmly.py
├── 5_Quint8_anmly_quantize.py
│
├── 6_test_anmly.py
├── 6_verify_anmly_static.py
│
├── 7_anmly_accuracy_report.py
├── 7_benchmark_anmly.py
│
└── README.md

```

---

# Step-by-Step Pipeline

## Step 1 — Understand the Anomaly Model Architecture

Script:

```

1_anmly_architecture.py

```

Purpose:

This script helps analyze the internal structure of the anomaly model.

It examines:

- Backbone feature extractor
- Embedding outputs
- Feature map layers used for anomaly scoring
- Memory bank / embedding dimensions

This step helps understand **which layers should or should not be quantized**.

Example:

```

python3 1_anmly_architecture.py --model_path MODEL_PATH

```

---

# Step 2 — Inspect the ONNX Model

Script:

```

1_check_anomoly_model.py

```

Purpose:

Prints detailed information about the ONNX model including:

- Model input names
- Input tensor shapes
- Output nodes
- Operator types
- Total number of nodes
- ONNX opset version

This step ensures the model is ready for quantization.

Example:

```

python3 1_check_anomoly_model.py --model_path MODEL_PATH

```

---

# Step 3 — Check Which Layers Are Quantized

Script:

```

1_check_what_quantized.py

```

Purpose:

After quantization, this script identifies:

- Which nodes are quantized
- Locations of QuantizeLinear / DequantizeLinear nodes
- Layers still running in FP32

This helps verify whether quantization was applied correctly.

Example:

```

python3 1_check_what_quantized.py --model_path MODEL_PATH

```

---

# Step 4 — Find Quantization-Sensitive Nodes

Script:

```

2_find_nodes.py

```

Purpose:

Detects nodes that may cause problems during INT8 quantization.

This script:

- Lists all operators in the graph
- Identifies nodes that should be excluded from quantization
- Helps avoid runtime errors

Example:

```

python3 2_find_nodes.py --model_path MODEL_PATH

```

---

# Step 5 — Upgrade ONNX Opset

Script:

```

3_upgrade_cls_opset.py

```

Purpose:

Upgrades the model to a **newer ONNX opset version (typically opset 17)**.

Why this is necessary:

Older opsets often produce errors during quantization such as:

```

DequantizeLinear not supported

```

Example:

```

python3 3_upgrade_cls_opset.py --model_path MODEL_PATH

```

Output:

```

model_opset17.onnx

```

---

# Step 6 — Model Preprocessing

Script:

```

4_preprocess_cls_model.py

```

Purpose:

Runs **ONNX shape inference** and graph preprocessing.

This ensures:

- All tensor shapes are defined
- The model graph is ready for quantization

Example:

```

python3 4_preprocess_cls_model.py --model_path model_opset17.onnx

```

Output:

```

model_preprocessed.onnx

```

---

# Step 7 — Quantization Experiments

This folder contains **three quantization approaches**.

---

## 7.1 QDQ Static Quantization

Script:

```

5_QDQ_quantize_256_anmly_static.py

```

This method uses:

- Static calibration
- QDQ format
- Calibration image dataset

Benefits:

- Best accuracy preservation
- Recommended for production

Example:

```

python3 5_QDQ_quantize_256_anmly_static.py 
--model_path model_preprocessed.onnx 
--calib_dir calibration_images/

```

---

## 7.2 QINT8 Quantization

Script:

```

5_Qint8_quantize_anmly.py

```

Quantizes tensors using **signed INT8 format**.

Range:

```

-128 to 127

```

Suitable for:

- Symmetric quantization
- Some CPU accelerators

---

## 7.3 QUINT8 Quantization

Script:

```

5_Quint8_anmly_quantize.py

```

Quantizes tensors using **unsigned INT8 format**.

Range:

```

0 to 255

```

Often used for:

- Asymmetric quantization
- ONNX Runtime CPU inference

---

# Step 8 — Test Quantized Model

Script:

```

6_test_anmly.py

```

Purpose:

Runs inference on the quantized model using test images.

It verifies:

- Model execution
- Output anomaly score
- Detection results

---

# Step 9 — Verify Static Quantized Model

Script:

```

6_verify_anmly_static.py

```

Ensures the static quantized model:

- Loads correctly in ONNX Runtime
- Produces valid outputs
- Matches expected tensor shapes

---

# Step 10 — Accuracy Validation

Script:

```

7_anmly_accuracy_report.py

```

Compares anomaly detection performance between:

- FP32 model
- INT8 model

Metrics may include:

- anomaly score difference
- decision threshold comparison
- anomaly detection consistency

Example:

```

python3 7_anmly_accuracy_report.py 
--fp32_model model_fp32.onnx 
--int8_model model_int8.onnx

```

---

# Step 11 — Benchmark Performance

Script:

```

7_benchmark_anmly.py

```

Measures inference speed for:

- FP32 model
- INT8 model

Example output:

| Model | Latency |
|------|--------|
| FP32 | 95 ms |
| INT8 | 58 ms |

This helps evaluate performance improvements.

---

# Recommended Execution Order

Run the scripts in this order:

```

1_anmly_architecture.py
1_check_anomoly_model.py
1_check_what_quantized.py
2_find_nodes.py
3_upgrade_cls_opset.py
4_preprocess_cls_model.py
5_QDQ_quantize_256_anmly_static.py
6_verify_anmly_static.py
6_test_anmly.py
7_anmly_accuracy_report.py
7_benchmark_anmly.py

```

---

# Notes

- Use **50–200 calibration images** for stable quantization
- Avoid quantizing embedding layers if accuracy drops
- QDQ quantization generally provides the best balance between **performance and accuracy**

---

