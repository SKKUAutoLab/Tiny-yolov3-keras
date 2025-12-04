# SKKU Automation LAB
# YOLOv3-Tiny Vitis-AI Pipeline (Training, Quantization, Compilation)

This repository provides an end-to-end implementation of training, quantization, and compilation for the YOLOv3-Tiny model within the Vitis-AI Docker environment.
It is the updated package for the 2026 Winter Extracurricular Program – Autonomous Driving SoC Design, integrating all steps into a single project.
Each participant prepares their own dataset and runs training with their own data.

Compared to v1.0 (2025 Winter Program), this version:
- Integrates training, quantization, and compilation in one repository
- Simplifies the workflow around a single YOLOv3-Tiny configuration
- Provides a cleaner directory structure for practical use

--------------------------------------------------------------------

## Repository Layout

vitis-ai-yolov3-tiny/
│── arch.json
│── train_yolov3.py
│── eval_yolov3tiny.py
│── compile_B1600_tiny-yolov3.sh
│
│── cfg/
│── common/
│── configs/
│── data/
│── float/
│── logs/
│── tools/
│── yolo3/
└── (other support files)

- train_yolov3.py               : Training script for YOLOv3-Tiny
- eval_yolov3tiny.py            : Evaluation / quantization entry script
- compile_B1600_tiny-yolov3.sh  : Vitis-AI compiler script for B1600 DPU
- arch.json                     : Architecture description for Vitis-AI compiler

--------------------------------------------------------------------
## All commands below assume the user is inside the container:

cd vitis-ai-yolov3-tiny

## Entering the Vitis-AI Docker Environment

Use the provided Docker launcher script:

./docker_run.sh cpu
# or
./docker_run.sh gpu

--------------------------------------------------------------------

## Training

Participants prepare their own dataset and annotations, then run:

python train_yolov3.py \
  --anchors_path configs/tiny_yolo3_anchors.txt \
  --classes_path configs/lane_class.txt \
  --annotation_file data/lane_detection/train/_annotations.txt \
  --model_input_shape 256x256 \
  --batch_size 16

Training outputs (checkpoints and logs) are saved in:

logs/yolov3_tiny/

Examples:
logs/yolov3_tiny/epXXX-lossXX-val_lossXX.h5

--------------------------------------------------------------------

## Quantization and Evaluation

python eval_yolov3tiny \
  --model_path logs/yolov3_tiny/epXXX-lossXX-val_lossXX.h5 \
  --anchors_path configs/tiny_yolo3_anchors.txt \
  --classes_path configs/lane_class.txt \
  --annotation_file data/lane_detection/train/_annotations.txt \
  --quantize \
  --eval_quant

This step:
- Loads the trained floating-point YOLOv3-Tiny model
- Performs Vitis-AI quantization
- Evaluates the quantized model if requested

Quantized model files and outputs are generated according to internal script configuration.

--------------------------------------------------------------------

## Compilation (Vitis-AI Compiler for B1600)

./compile_B1600_tiny-yolov3.sh

The compiler script uses:
- arch.json (DPU architecture description)
- Quantized YOLOv3-Tiny model

and produces:
- Compiled xmodel for deployment on B1600 DPU

Output directory and filenames are configured inside the script.

--------------------------------------------------------------------

## Summary

Training:         train_yolov3.py
Quantization:     eval_yolov3tiny (with --quantize --eval_quant)
Compilation:      compile_B1600_tiny-yolov3.sh

This repository packages the full YOLOv3-Tiny optimization pipeline for Vitis-AI in a single, Docker-based workflow.
