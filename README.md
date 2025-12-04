# 🚀 YOLOv3-Tiny Vitis-AI Pipeline (End-to-End Workflow)

> 💡 **Official Repository for the 2026 Winter Extracurricular Program – Autonomous Driving SoC Design**  
>  
> This project provides a fully integrated workflow for **Training → Quantization → Compilation** of the **YOLOv3-Tiny** model inside the Vitis-AI environment.  
>  
> **Participants must prepare their own datasets**, then follow this pipeline to optimize and compile the model for deployment on the **B1600 DPU**.

---

## ✨ Key Features (v2.0)

- **Integrated Workflow**: Training, quantization, and compilation all handled within a single repository  
- **Simplified Configuration**: Streamlined around the YOLOv3-Tiny model  
- **Practical Structure**: Clean and compact layout optimized for hands-on exercises  

---

## 📂 Repository Layout

The following is the main file and directory structure of the project root.  
(Displayed in a code block to preserve the tree structure.)

```text
vitis-ai-yolov3-tiny/
├── arch.json                       # DPU Compiler Architecture Definition
├── train_yolov3.py                 # 1️⃣ Training script (Generates Float Model)
├── eval_yolov3tiny.py              # 2️⃣ Evaluation/Quantization script (Generates Quantized Model)
├── compile_B1600_tiny-yolov3.sh    # 3️⃣ Compilation script (Generates xmodel)
│
├── cfg/                            # YOLOv3-Tiny configuration files
├── common/                         # Common utility code
├── configs/                        # Anchor and class definition files
├── data/                           # User datasets and annotation files
├── float/                          # Intermediate files for quantization (optional)
├── logs/                           # Training checkpoints and logs
├── tools/                          # Auxiliary tools
├── yolo3/                          # YOLOv3 model core code
└── (other support files)
File Name	Description
train_yolov3.py	Training entry script for generating the Floating-Point Model
eval_yolov3tiny.py	Converts & evaluates the trained model into a Quantized Model
compile_B1600_tiny-yolov3.sh	Compiles the Quantized Model into a DPU-ready xmodel
arch.json	DPU architecture configuration for the Vitis-AI compiler

🛠️ Getting Started
💬 All commands are executed inside the Vitis-AI Docker container.

1. 🐳 Entering the Docker Environment
Run the Vitis-AI container using the provided launcher script:

bash
코드 복사
./docker_run.sh cpu
# or, for GPU support:
./docker_run.sh gpu
Navigate into the project directory after entering the container:

bash
코드 복사
cd vitis-ai-yolov3-tiny
2. 🚂 Training
Place your dataset and annotation files under the data/ directory, then start training.

Objective: Generate a floating-point YOLOv3-Tiny model

Example output: epXXX-lossXX-val_lossXX.h5

Output Directory: logs/yolov3_tiny/

bash
코드 복사
python train_yolov3.py \
    --anchors_path configs/tiny_yolo3_anchors.txt \
    --classes_path configs/lane_class.txt \
    --annotation_file data/lane_detection/train/_annotations.txt \
    --model_input_shape 256x256 \
    --batch_size 16
3. 📉 Quantization & Evaluation
Convert the trained Floating-Point model (logs/yolov3_tiny/*.h5) into a Quantized Model optimized for the DPU.

Objective: Produce the quantized model file & evaluate accuracy

Key Options

--quantize : Perform quantization

--eval_quant : Evaluate the quantized model

bash
코드 복사
python eval_yolov3tiny.py \
    --model_path logs/yolov3_tiny/epXXX-lossXX-val_lossXX.h5 \
    --anchors_path configs/tiny_yolo3_anchors.txt \
    --classes_path configs/lane_class.txt \
    --annotation_file data/lane_detection/train/_annotations.txt \
    --quantize \
    --eval_quant
4. ⚙️ Compilation (for B1600 DPU)
Compile the Quantized Model into the final xmodel format for deployment on the B1600 DPU.

Objective: Generate the DPU-executable xmodel

Inputs

arch.json

Quantized model produced in step 3

Output Location: Defined inside the compilation script

bash
코드 복사
./compile_B1600_tiny-yolov3.sh
📌 Pipeline Summary
Step	Script	Output
Training	train_yolov3.py	Floating-point .h5 checkpoint
Quantization	eval_yolov3tiny.py	Quantized Model file
Compilation	compile_B1600_tiny-yolov3.sh	DPU-ready xmodel
