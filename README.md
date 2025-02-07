# SKKU-Tiny-yolov3-keras Tutorial

> **YOLOv3-tiny Tutorial**  
> - 커스텀 이미지 감지 데이터 학습을 위한 사전 학습된 모델 가중치와 함께  
>   Keras와 TensorFlow 백엔드를 활용한 YOLOv3-tiny 모델 구현체

---

## 학습 목표

- **Roboflow**를 활용해 커스텀 이미지 감지 데이터 불러오기
- **Keras**에서 YOLOv3-tiny 모델 설정하기
- YOLOv3-tiny 모델 학습 진행하기
- 학습된 모델을 이용한 추론(Inference) 수행하기
- 향후 양자화 및 FPGA 컴파일을 위한 Keras 모델 가중치 저장하기

---

## 모델 가중치 다운로드

사전 학습된 모델 가중치를 아래 링크에서 다운로드 받을 수 있습니다:

[**YOLOv3 모델 가중치 다운로드**](https://drive.google.com/uc?id=1Ybgwyc57cBnq9Byo41zuzOmBdFcRWNRL)

---

## Roboflow를 이용한 데이터 관리

[Roboflow](https://roboflow.ai)는 컴퓨터 비전 데이터셋을 관리, 전처리, 증강, 버전 관리하는 데 최적화된 도구입니다.

**주요 이점:**
- 코드 작성량 50% 감소
- 어노테이션 품질 자동 확인
- 학습 시간 단축
- 모델 재현성 향상

> **예시 이미지:**
> 
> ![Roboflow 예시 이미지](https://i.imgur.com/WHFqYSJ.png)

---

[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

---

이 저장소는 **Keras**를 활용한 **YOLOv3-tiny** 구현체로,  
기존 [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K)를 참고하여 작성되었습니다.
