```markdown
# ONNX-INT8 Quantization Pipeline

Automated INT8 quantization pipeline for **Segmentation, Classification and Anomaly detection models** using **ONNX Runtime QDQ static quantization**.

The repository provides both:

- Step-by-step scripts for understanding each stage of quantization
- Unified pipeline for fully automated quantization from terminal

---

# Features

✔ Automatic ONNX model inspection  
✔ Detect incompatible nodes for INT8 quantization  
✔ Automatic opset upgrade (fixes DequantizeLinear errors)  
✔ Shape inference preprocessing  
✔ QDQ static INT8 quantization  
✔ Calibration using real images  
✔ FP32 vs INT8 benchmarking  
✔ Accuracy validation  

---

# Repository Structure

```

```
onnx-int8-quantization-pipeline/
│
├── README.md
├── accuracy_report.py
│
├── segmentation/
│   ├── README.md
│   ├── 1_inspect_model.py
│   ├── 2_find_nodes.py
│   ├── 3_upgrade_opset.py
│   ├── 4_preprocess_model.py
│   ├── 5_quantize_int8.py
│   └── 6_validate_benchmark.py
│
├── classification/
│   ├── README.md
│   ├── 1_inspect_model.py
│   ├── 2_find_nodes.py
│   ├── 3_upgrade_opset.py
│   ├── 4_preprocess_model.py
│   ├── 5_quantize_int8.py
│   └── 6_validate_benchmark.py
│
├── anomaly/
│   ├── README.md
│   ├── 1_inspect_model.py
│   ├── 2_find_nodes.py
│   ├── 3_upgrade_opset.py
│   ├── 4_preprocess_model.py
│   ├── 5_quantize_int8.py
│   └── 6_validate_benchmark.py
│
└── unified_pipeline/
    ├── README.md
    └── quantize_pipeline.py
```

---

# Installation

Install required dependencies:

```bash
pip install onnx onnxruntime opencv-python numpy
````

Or activate the existing environment :
```bash
source ~/onnx_env/bin/activate
```
---

# Unified Quantization Pipeline

The unified pipeline automatically performs the following steps:

1. Environment validation
2. Model inspection
3. Incompatible node detection
4. Opset upgrade (if required)
5. Shape inference preprocessing
6. Static INT8 quantization using QDQ format
7. Benchmark comparison (FP32 vs INT8)
8. Accuracy validation

---

# Example Usage

## Segmentation Model

```bash
python3 quantize_pipeline.py \
    --model_path /home/qualviz/QualViz/models/512_semsons.onnx \
    --model_type segmentation \
    --calib_dir /home/qualviz/calibration_images/
```

---

## Classification Model

```bash
python3 quantize_pipeline.py \
    --model_path /home/qualviz/QualViz/models/Classification/512_2_clstalbros.onnx \
    --model_type classification \
    --calib_dir /home/qualviz/calibration_images/
```

---

## Anomaly Model (PatchCore)

```bash
python3 quantize_pipeline.py \
    --model_path /home/qualviz/QualViz/models/anomaly/256_anomaly.onnx \
    --model_type anomaly \
    --calib_dir /home/qualviz/calibration_images/
```

---

## PaDiM Anomaly Model

```bash
python3 quantize_pipeline.py \
    --model_path /home/qualviz/QualViz/models/padim_anmly/Padim.onnx \
    --model_type anomaly \
    --calib_dir /home/qualviz/QualViz/calibration_images/anomaly/ \
    --output_dir /home/qualviz/QualViz/models/padim_anmly/
```

---

# Benchmark Results

Example performance comparison:

| Model          | FP32 (ms) | INT8 (ms) | Speedup |
| -------------- | --------- | --------- | ------- |
| Segmentation   | 120       | 70        | 1.7x    |
| Classification | 35        | 21        | 1.6x    |
| Anomaly        | 95        | 58        | 1.6x    |

---

# Accuracy Validation

The pipeline performs validation to ensure INT8 accuracy remains close to FP32.

Validation methods include:

* **Segmentation:** Pixel-wise output comparison
* **Classification:** Top-1 class match check
* **Anomaly Detection:** Decision threshold consistency

---

# License

MIT License

```
