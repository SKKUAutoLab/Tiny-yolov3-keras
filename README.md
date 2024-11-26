# SKKU-Tiny-yolov3-keras tutorial

## 학습 목표
* Roboflow에서 커스텀 이미지 감지 데이터를 불러오는 방법
* Keras에서 YOLOv3-tiny 모델을 설정하는 방법
* YOLOv3-tiny 모델을 학습시키는 방법
* 학습된 모델을 이용하여 추론(Inference)하는 방법
* 추후 양자화 및 FPGA 컴파일을 위해 Keras 모델 가중치를 저장하는 방법

## 모델 가중치 다운로드

이 저장소를 사용하는 데 도움이 되도록 사전 학습된 모델 가중치를 제공합니다. 아래 링크에서 가중치를 다운로드할 수 있습니다:

**[YOLOv3 모델 가중치 다운로드](https://drive.google.com/uc?id=1Ybgwyc57cBnq9Byo41zuzOmBdFcRWNRL)**

## Roboflow를 이용한 데이터 관리

[Roboflow](https://roboflow.ai)는 컴퓨터 비전용 데이터셋을 관리, 전처리, 증강, 버전 관리하는 작업을 간단하게 만들어줍니다.  
개발자는 Roboflow를 활용해 코드 작성량을 50% 줄이고, 어노테이션 품질을 자동으로 확인하며, 학습 시간을 단축하고, 모델 재현성을 높일 수 있습니다.

![alt text](https://i.imgur.com/WHFqYSJ.png)

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

이 저장소는 Keras를 활용한 YOLOv3-tiny (TensorFlow 백엔드)의 구현체로, [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K)를 참고하여 작성되었습니다.
