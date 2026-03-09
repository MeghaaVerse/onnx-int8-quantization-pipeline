Classification INT8 Quantization Pipeline
=========================================

This folder contains a **step-by-step pipeline to convert an ONNX Classification model from FP32 to INT8** using **ONNX Runtime static QDQ quantization**.

The scripts allow you to **inspect, prepare, quantize, and validate classification models** while ensuring minimal accuracy loss.

Each script represents **one stage of the quantization process**.

Pipeline Overview
=================

The classification quantization workflow consists of the following stages:

1.  Model inspection
    
2.  Node analysis
    
3.  Opset upgrade
    
4.  Shape inference preprocessing
    
5.  Static INT8 quantization
    
6.  Model verification
    
7.  Accuracy comparison
    

Folder Structure
================

```
classification/
│
├── 1_cls_architecture.py
├── 1_check_classification_model.py
├── 1_check_what_quantized.py
├── 2_find_nodes.py
├── 3_upgrade_cls_opset.py
├── 4_preprocess_cls_model.py
├── 5_QDQ_quantize_256_cls_static.py
├── 5_Qint8_quantize_cls.py
├── 5_Quint8_cls_quantize.py
├── 6_test_cls.py
├── 6_verify_cls_static.py
├── 7_cls_accuracy_report.py
└── 7_benchmark_cls.py
``` 

Script Descriptions
===================

### 1\_cls\_architecture.py

Analyzes the architecture of the classification model to understand layer composition and network structure.

### 1\_check\_classification\_model.py

Loads the ONNX classification model and verifies that it runs correctly using ONNX Runtime.

### 1\_check\_what\_quantized.py

Inspects the model graph to determine which layers are quantized after INT8 conversion.

### 2\_find\_nodes.py

Lists all nodes in the ONNX graph and identifies layers that may require special handling during quantization.

### 3\_upgrade\_cls\_opset.py

Upgrades the ONNX model to a newer opset version to ensure compatibility with quantization tools.

### 4\_preprocess\_cls\_model.py

Applies preprocessing steps such as shape inference and graph cleanup to prepare the model for quantization.

### 5\_QDQ\_quantize\_256\_cls\_static.py

Performs **static QDQ INT8 quantization** using calibration images of size **256×256**.

### 5\_Qint8\_quantize\_cls.py

Performs **QINT8 quantization** for experimentation and compatibility testing.

### 5\_Quint8\_cls\_quantize.py

Applies **QUINT8 quantization** for additional quantization comparison.

### 6\_test\_cls.py

Runs inference tests on the quantized model to ensure predictions are generated correctly.

### 6\_verify\_cls\_static.py

Validates the static INT8 model outputs against the original FP32 model.

### 7\_cls\_accuracy\_report.py

Generates an **accuracy comparison report** between FP32 and INT8 classification results.

### 7\_benchmark\_cls.py

Benchmarks the quantized model to evaluate **inference speed improvements**.

Expected Output
===============

After running the pipeline, you should obtain:

*   INT8 quantized classification ONNX models
    
*   FP32 vs INT8 accuracy comparison reports
    
*   Inference benchmarking results
    
*   Quantization validation logs
    

Notes
=====

*   Always verify classification accuracy after quantization.
    
*   Calibration data should represent the **training data distribution**.
    
*   Some sensitive layers may remain in **FP32** to maintain model accuracy.
    

License
=======

This project is licensed under the **MIT License**.
