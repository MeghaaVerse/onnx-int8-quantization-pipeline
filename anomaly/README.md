Anomaly Detection INT8 Quantization Pipeline
============================================

This folder contains scripts to **analyze, quantize, and validate ONNX anomaly detection models** such as **PatchCore and PaDiM** using **ONNX Runtime INT8 quantization**.

Unlike classification or segmentation, anomaly models rely heavily on **feature embeddings and distance calculations**, so quantization must be performed carefully to avoid degrading anomaly detection accuracy.

This folder includes **multiple quantization experiments**:

*   QDQ static quantization
    
*   QINT8 quantization
    
*   QUINT8 quantization
    

Pipeline Overview
=================

The anomaly quantization workflow includes the following stages:

1.  Understanding anomaly model architecture
    
2.  Inspecting the ONNX model
    
3.  Checking which layers are quantized
    
4.  Detecting unsupported nodes
    
5.  Upgrading ONNX opset
    
6.  Model preprocessing with shape inference
    
7.  INT8 quantization experiments
    
8.  Model verification and testing
    
9.  Benchmark comparison
    
10.  Accuracy validation
    

Folder Structure
================
```
anomaly/
│
├── 1_anmly_architecture.py
├── 1_check_anomoly_model.py
├── 1_check_what_quantized.py
├── 2_find_nodes.py
├── 3_upgrade_cls_opset.py
├── 4_preprocess_cls_model.py
├── 5_QDQ_quantize_256_anmly_static.py
├── 5_Qint8_quantize_anmly.py
├── 5_Quint8_anmly_quantize.py
├── 6_test_anmly.py
├── 6_verify_anmly_static.py
├── 7_anmly_accuracy_report.py
└── 7_benchmark_anmly.py
```

Script Descriptions
===================

### 1\_anmly\_architecture.py

Analyzes the architecture of the anomaly detection model to understand its layer composition and feature extraction pipeline.

### 1\_check\_anomoly\_model.py

Loads the ONNX anomaly model and checks whether it can be executed correctly using ONNX Runtime.

### 1\_check\_what\_quantized.py

Inspects the ONNX model and determines which layers have been quantized after INT8 conversion.

### 2\_find\_nodes.py

Identifies specific nodes within the ONNX graph that may require attention before quantization.

### 3\_upgrade\_cls\_opset.py

Upgrades the ONNX model to a newer opset version to ensure compatibility with quantization tools.

### 4\_preprocess\_cls\_model.py

Applies preprocessing steps such as shape inference and graph optimization to prepare the model for quantization.

### 5\_QDQ\_quantize\_256\_anmly\_static.py

Performs **QDQ static INT8 quantization** using calibration data.

### 5\_Qint8\_quantize\_anmly.py

Applies **QINT8 quantization** to the anomaly detection model.

### 5\_Quint8\_anmly\_quantize.py

Applies **QUINT8 quantization** for experimentation and performance comparison.

### 6\_test\_anmly.py

Runs inference tests to verify whether the quantized model is producing outputs correctly.

### 6\_verify\_anmly\_static.py

Validates the static quantized model against the original FP32 model.

### 7\_anmly\_accuracy\_report.py

Generates accuracy reports comparing **FP32 vs INT8 anomaly detection performance**.

### 7\_benchmark\_anmly.py

Benchmarks inference performance and compares **latency improvements after quantization**.

Expected Output
===============

After running the pipeline you should obtain:

*   Quantized anomaly detection ONNX models
    
*   Accuracy comparison reports
    
*   Inference benchmark results
    
*   Quantization verification logs
    

Notes
=====

*   Always validate anomaly detection accuracy after quantization.
    
*   Some layers may need to remain in **FP32** to maintain detection quality.
    
*   Calibration data should represent **normal samples** from the dataset.
    

License
=======

This project is licensed under the **MIT License**.
