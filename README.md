# 🚀 YOLOv3-Tiny Vitis-AI Pipeline (End-to-End Workflow)

> 💡 Official Repository for the **2026 Winter Extracurricular Program – Autonomous Driving SoC Design**
>
> This project provides an integrated, end-to-end implementation of **Training, Quantization, and Compilation** for the **YOLOv3-Tiny** model within the Vitis-AI environment.
>
> **Participants are expected to prepare their own dataset** and follow this pipeline to optimize and compile the model for the B1600 DPU.

## ✨ Key Features (v2.0)

* **Integrated Workflow**: The entire process—training, quantization, and compilation—is contained within a single repository.
* **Simplified Configuration**: The workflow is streamlined around a single YOLOv3-Tiny model configuration.
* **Practical Structure**: Provides a clean and concise directory structure optimized for practical exercises.

---

## 📂 Repository Layout

The core file and folder structure in the project root directory is as follows. The directory structure is enclosed in a code block for correct rendering in Markdown.

vitis-ai-yolov3-tiny/ ├── arch.json # DPU Compiler Architecture Definition File ├── train_yolov3.py # 1️⃣ Training script (Generates Float Model) ├── eval_yolov3tiny.py # 2️⃣ Evaluation/Quantization script (Generates Quantized Model) ├── compile_B1600_tiny-yolov3.sh # 3️⃣ Compilation script (Generates xmodel) │ ├── cfg/ # YOLOv3-Tiny configuration files ├── common/ # Common utility code ├── configs/ # Anchor and class definition files ├── data/ # Location for user datasets and annotation files ├── float/ # Intermediate files for quantization (optional) ├── logs/ # Training checkpoints and logs storage ├── tools/ # Auxiliary tools ├── yolo3/ # YOLOv3 model core code └── (other support files)


| File Name | Role |
| :--- | :--- |
| **`train_yolov3.py`** | Entry script for generating the **Floating Point Model**. |
| **`eval_yolov3tiny.py`** | Converts and evaluates the trained model into a **Quantized Model**. |
| **`compile_B1600_tiny-yolov3.sh`** | Final compilation of the Quantized Model into an **xmodel** for the B1600 DPU. |
| **`arch.json`** | DPU architecture configuration used by the Vitis-AI compiler. |

---

## 🛠️ Getting Started: The Workflow

All commands assume the user is executing them **inside** the Vitis-AI Docker container.

### 1. 🐳 Entering the Docker Environment

Run the Vitis-AI container using the provided launcher script.

```bash
./docker_run.sh cpu
# or if using GPU
./docker_run.sh gpu

# After entering the container, navigate to the project directory
cd vitis-ai-yolov3-tiny
2. 🚂 Training
Participants prepare their dataset and annotation files in the data/ path, then start training.

Objective: Generate a floating-point YOLOv3-Tiny checkpoint (epXXX-lossXX-val_lossXX.h5).

Output: Saved in the logs/yolov3_tiny/ directory.

Bash

python train_yolov3.py \
    --anchors_path configs/tiny_yolo3_anchors.txt \
    --classes_path configs/lane_class.txt \
    --annotation_file data/lane_detection/train/_annotations.txt \
    --model_input_shape 256x256 \
    --batch_size 16
3. 📉 Quantization and Evaluation
The trained floating-point model (logs/yolov3_tiny/*.h5) is converted into an integer (Quantized) model optimized for the DPU using Vitis-AI.

Objective: Generate the Quantized Model file and evaluate its accuracy.

Key Options: --quantize (performs quantization), --eval_quant (evaluates the quantized model).

Bash

python eval_yolov3tiny.py \
    --model_path logs/yolov3_tiny/epXXX-lossXX-val_lossXX.h5 \
    --anchors_path configs/tiny_yolo3_anchors.txt \
    --classes_path configs/lane_class.txt \
    --annotation_file data/lane_detection/train/_annotations.txt \
    --quantize \
    --eval_quant
4. ⚙️ Compilation (for B1600 DPU)
The quantized model is compiled into the final executable file, xmodel, ready for deployment on the B1600 DPU.

Objective: Generate the DPU-ready xmodel file.

Input: arch.json and the Quantized Model file.

Output: The xmodel file is generated in the path defined inside the script.

Bash

./compile_B1600_tiny-yolov3.sh
📌 Pipeline Summary
Step	Script	Main Artifact
Training	train_yolov3.py	Floating-point .h5 checkpoint
Quantization	eval_yolov3tiny.py	Quantized Model file
Compilation	compile_B1600_tiny-yolov3.sh	xmodel for DPU Deployment
