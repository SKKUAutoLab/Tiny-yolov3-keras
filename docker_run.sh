#!/bin/bash

########################################
# AutomationLab VisionHW
########################################

CPU_IMAGE="xilinx/vitis-ai-cpu"
GPU_IMAGE="joocoo/vitis-ai-gpu:2.5"

# 1) 이미지 선택 (arg or 메뉴)
if [ -n "$1" ]; then
    case "$1" in
        cpu|CPU)
            IMAGE="$CPU_IMAGE"
            ;;
        gpu|GPU)
            IMAGE="$GPU_IMAGE"
            ;;
        *)
            IMAGE="$1"
            ;;
    esac
else
    echo "==============================="
    echo " Vitis-AI Docker 실행 모드 선택"
    echo "==============================="
    echo "  1) CPU  이미지 : $CPU_IMAGE"
    echo "  2) GPU  이미지 : $GPU_IMAGE"
    echo
    read -p "선택 [1/2] (기본: 1) : " CHOICE

    case "$CHOICE" in
        2)
            IMAGE="$GPU_IMAGE"
            ;;
        *)
            IMAGE="$CPU_IMAGE"
            ;;
    esac
fi

echo ">> Docker Image: $IMAGE"
echo ">> Host Workspace: $PWD"

# 2) 현재 디렉토리(PWD)를 workspace로 사용
WORKSPACE="$PWD"

# 3) GPU 플래그
GPU_FLAG=""
if [[ "$IMAGE" == "$GPU_IMAGE" ]]; then
    GPU_FLAG="--gpus all"
fi

# 4) Docker 실행
docker run --rm -it \
    $GPU_FLAG \
    -e UID=$(id -u) -e GID=$(id -g) \
    -v "$WORKSPACE":/workspace \
    -v /opt/Vitis-AI:/opt/Vitis-AI:ro \
    -w /workspace \
    --name "vitis-ai-$(whoami)" \
    "$IMAGE" \
    /bin/bash
