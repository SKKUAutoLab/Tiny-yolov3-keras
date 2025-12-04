# YOLOv3-Tiny Vitis-AI Pipeline (End-to-End Workflow)

## Repository Layout

```text
vitis-ai-yolov3-tiny/
├── arch.json
├── train_yolov3.py
├── eval_yolov3tiny.py
├── compile_B1600_tiny-yolov3.sh
│
├── cfg/
├── common/
├── configs/
├── data/
├── float/
├── logs/
├── tools/
├── yolo3/
└── (other support files)
```

## Entering the Docker Environment

```bash
./docker_run.sh cpu
```

```bash
./docker_run.sh gpu
```

```bash
cd vitis-ai-yolov3-tiny
```

## Training

```bash
python train_yolov3.py   --anchors_path configs/tiny_yolo3_anchors.txt   --classes_path configs/lane_class.txt   --annotation_file data/lane_detection/train/_annotations.txt   --model_input_shape 256x256   --batch_size 16
```

## Quantization & Evaluation

```bash
python eval_yolov3tiny.py   --model_path logs/yolov3_tiny/epXXX-lossXX-val_lossXX.h5   --anchors_path configs/tiny_yolo3_anchors.txt   --classes_path configs/lane_class.txt   --annotation_file data/lane_detection/train/_annotations.txt   --quantize   --eval_quant
```

## Compilation

```bash
./compile_B1600_tiny-yolov3.sh
```

## Pipeline Summary

| Step | Script | Artifact |
|------|--------|----------|
| Training | train_yolov3.py | Float Model (.h5) |
| Quantization | eval_yolov3tiny.py | Quantized Model |
| Compilation | compile_B1600_tiny-yolov3.sh | xmodel |
