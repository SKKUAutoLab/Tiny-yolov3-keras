#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiny YOLOv3 Training Script (Darknet backbone, lane detection)

- 모델 타입: tiny_yolo3_darknet (고정)
- Dataset: lane_detection (annotation txt)
- Anchors: tiny_yolo3_anchors.txt (6 anchors)
- Classes: lane_class.txt

예시:
  python train_yolov3.py \
      --anchors_path configs/tiny_yolo3_anchors.txt \
      --classes_path configs/lane_class.txt \
      --annotation_file data/lane_detection/train/_annotations.txt \
      --model_input_shape 256x256 \
      --batch_size 16
"""

import os

# ===== 환경 세팅 =====
# NOTE: 이 값들은 반드시 tensorflow import "이전"에 설정돼야 한다.
#       (예전 코드는 import 뒤에 설정해서 전부 무효였다)
# TF_ENABLE_AUTO_MIXED_PRECISION / GRAPH_REWRITE_* 는 TF 1.x grappler 전용
# 플래그라 TF 2.x 에서는 아무 효과가 없어 제거했다. TF 2.x 에서 mixed precision
# 이 필요하면 tf.keras.mixed_precision.set_global_policy('mixed_float16') 를
# 써야 하지만, DPU 양자화 흐름과 충돌할 수 있어 기본으로 켜지 않는다.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import argparse
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import (
    TensorBoard, ModelCheckpoint, ReduceLROnPlateau,
    EarlyStopping, TerminateOnNaN, Callback
)
from tensorflow.keras.models import Model

from yolo3.model import get_yolo3_train_model, build_adam
from yolo3.data import yolo3_data_generator_wrapper
from common.utils import get_classes, get_anchors, get_dataset, optimize_tf_gpu
from common.callbacks import CheckpointCleanCallBack

optimize_tf_gpu(tf, K)


# =====================================================
# 1. 커스텀 콜백: EpochTracker & BestMetricCheckpoint
# =====================================================
class EpochTracker(Callback):
    """마지막으로 끝난 epoch 번호(1-based, 절대값)를 추적.

    Keras 가 넘겨주는 `epoch` 은 initial_epoch 을 포함한 절대 인덱스라
    직접 카운트하지 않고 그대로 쓰면 재개(resume)/2단계 학습에서도 정확하다.
    """
    def __init__(self, init_epoch=0):
        super().__init__()
        self.epoch = init_epoch  # 아직 아무 epoch 도 안 끝났으면 init_epoch

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch + 1


class BestMetricCheckpoint(Callback):
    """
    monitor(metric) 기준으로 가장 좋은 모델을 항상 따로 저장하는 콜백.
    - mode: 'min' (loss), 'max' (mAP) 등 선택 가능
    - epoch 번호는 콜백에 전달된 `epoch` 인자에서 직접 뽑는다.
      (다른 콜백의 상태에 의존하면 콜백 실행 순서에 따라 off-by-one 이 난다)
    """
    def __init__(self, log_dir, monitor='val_loss', mode='min',
                 filename_prefix='best', save_weights_only=False):
        super().__init__()
        self.log_dir = log_dir
        self.monitor = monitor
        self.mode = mode
        self.filename_prefix = filename_prefix
        self.save_weights_only = save_weights_only
        self._warned_missing = False

        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")

        self.best = np.inf if mode == 'min' else -np.inf
        self.best_path = None
        self.best_epoch = None

    def is_better(self, current):
        if self.mode == 'min':
            return current < self.best
        return current > self.best

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor, None)
        if current is None:
            if not self._warned_missing:
                print(f"[WARN][BestMetricCheckpoint] monitor '{self.monitor}' 가 "
                      f"logs 에 없습니다 (available: {sorted(logs.keys())}). "
                      f"best 모델 저장이 비활성화됩니다.")
                self._warned_missing = True
            return

        global_epoch = epoch + 1  # 1-based, initial_epoch 포함

        if self.is_better(current):
            self.best = current
            filename = (f"{self.filename_prefix}_{self.monitor}"
                        f"_epoch{global_epoch:03d}_{current:.4f}.h5")
            save_path = os.path.join(self.log_dir, filename)
            if self.save_weights_only:
                self.model.save_weights(save_path)
            else:
                self.model.save(save_path)
            self.best_path = save_path
            self.best_epoch = global_epoch
            print(
                f"[BestMetricCheckpoint] New best {self.monitor}={current:.6f} "
                f"at epoch {global_epoch}, saved to {save_path}"
            )


# =====================================================
# 2. 유틸
# =====================================================
def build_inference_model(model_body, model_pruning=False):
    """model_body 로부터 순수 추론용 모델을 만든다 (yolo_loss Lambda 제외).

    학습 모델은 `Model([model_body.input, *y_true], model_loss)` 로 만들어져
    model_body 의 "레이어를 그대로 재사용하는" flat 모델이다. 따라서
    model.layers 안에서 중첩 Model 을 찾는 방식은 절대 성공하지 못한다.
    반드시 get_yolo3_train_model() 이 돌려준 model_body 를 써야 한다.
    """
    if model_body is None:
        return None

    if model_pruning:
        try:
            from tensorflow_model_optimization.sparsity import keras as sparsity
            model_body = sparsity.strip_pruning(model_body)
            print('[INFO] pruning wrapper stripped from inference model.')
        except Exception as e:
            print(f'[WARN] strip_pruning on model_body failed: {e}')

    return Model(
        inputs=model_body.input,
        outputs=model_body.output,
        name="tiny_yolov3_inference"
    )


def fit_model(model, train_gen, steps_per_epoch, val_gen, validation_steps,
              epochs, initial_epoch, callbacks):
    """model.fit() 래퍼.

    fit_generator() 는 TF 2.1 부터 deprecated 이고 Keras 3 에서 제거됐다.
    파이썬 제너레이터는 model.fit() 이 그대로 받는다. (Sequence 가 아니므로
    workers/use_multiprocessing 인자는 어차피 무시되어 넘기지 않는다)
    """
    kwargs = dict(
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=1,
    )
    if val_gen is not None:
        kwargs['validation_data'] = val_gen
        kwargs['validation_steps'] = validation_steps

    return model.fit(train_gen, **kwargs)


def make_stage_callbacks(log_dir, monitor, shared_callbacks, model_pruning):
    """스테이지마다 새로 만들어야 하는 콜백 + 공용 콜백을 합쳐서 반환.

    ReduceLROnPlateau / EarlyStopping / ModelCheckpoint 는 내부 상태(best, wait)
    를 가지므로 두 스테이지가 같은 인스턴스를 공유하면 동작이 헷갈린다.
    스테이지별로 새로 만들고, 누적 상태가 필요한 것(EpochTracker,
    BestMetricCheckpoint)만 공유한다.
    """
    logging = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        update_freq='epoch',   # 'batch' 는 배치마다 디스크 flush 라 매우 느리다
    )

    if monitor == 'val_loss':
        ckpt_pattern = 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    else:
        ckpt_pattern = 'ep{epoch:03d}-loss{loss:.3f}.h5'

    checkpoint = ModelCheckpoint(
        os.path.join(log_dir, ckpt_pattern),
        monitor=monitor,
        mode='min',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        save_freq='epoch'
    )

    reduce_lr = ReduceLROnPlateau(
        monitor=monitor, factor=0.5, mode='min',
        patience=10, verbose=1, cooldown=0, min_lr=1e-10
    )

    early_stopping = EarlyStopping(
        monitor=monitor, min_delta=0, patience=50, verbose=1, mode='min'
    )

    callbacks = [logging, checkpoint, reduce_lr, early_stopping, TerminateOnNaN()]

    if model_pruning:
        # tfmot 은 이 콜백이 없으면 fit() 시점에 에러를 낸다 (step counter 미갱신)
        from tensorflow_model_optimization.sparsity import keras as sparsity
        callbacks.append(sparsity.UpdatePruningStep())
        callbacks.append(sparsity.PruningSummaries(log_dir=log_dir))

    # 공용 콜백은 마지막에. CheckpointCleanCallBack 은 저장이 다 끝난 뒤
    # 돌아야 하므로 BestMetricCheckpoint 뒤에 온다.
    callbacks.extend(shared_callbacks)
    return callbacks


# =====================================================
# 3. Tiny YOLOv3 Training Main
# =====================================================
def main(args):
    # ----- 기본 설정 -----
    log_dir = os.path.join('logs', 'tiny_yolov3')
    os.makedirs(log_dir, exist_ok=True)

    class_names = get_classes(args.classes_path)
    num_classes = len(class_names)
    if num_classes == 0:
        raise ValueError(f'no classes found in {args.classes_path}')

    anchors = get_anchors(args.anchors_path)
    num_anchors = len(anchors)

    # tiny-yolov3는 6 anchors, 2 feature layers
    if num_anchors != 6:
        raise ValueError(
            f'Tiny YOLOv3는 6개의 anchor를 사용해야 합니다 '
            f'(got {num_anchors} from {args.anchors_path}).')

    # ----- 입력 크기 -----
    try:
        height, width = args.model_input_shape.split('x')
        input_shape = (int(height), int(width))
    except ValueError:
        raise ValueError(
            f"--model_input_shape 는 '<height>x<width>' 형식이어야 합니다 "
            f"(got '{args.model_input_shape}')")
    if input_shape[0] % 32 != 0 or input_shape[1] % 32 != 0:
        raise ValueError('model_input_shape는 32의 배수여야 합니다.')

    # ----- 데이터 로드 -----
    # get_dataset 은 로컬 RandomState 로 셔플하므로 전역 numpy RNG(=augmentation)
    # 를 오염시키지 않는다. --data_seed 를 주면 split 이 실행 간 재현된다.
    if args.val_annotation_file:
        train_dataset = get_dataset(args.annotation_file, shuffle=True,
                                    seed=args.data_seed)
        val_dataset = get_dataset(args.val_annotation_file, shuffle=False)
    else:
        if not 0.0 <= args.val_split < 1.0:
            raise ValueError('--val_split 은 [0.0, 1.0) 범위여야 합니다.')
        dataset = get_dataset(args.annotation_file, shuffle=True,
                              seed=args.data_seed)
        num_val = int(len(dataset) * args.val_split)
        num_train = len(dataset) - num_val
        train_dataset = dataset[:num_train]
        val_dataset = dataset[num_train:]

    num_train = len(train_dataset)
    num_val = len(val_dataset)

    if num_train == 0:
        raise ValueError(f'no training samples found in {args.annotation_file}')

    has_val = num_val > 0
    monitor = 'val_loss' if has_val else 'loss'
    if not has_val:
        print('[WARN] validation set 이 비어 있습니다. checkpoint / early stopping / '
              "LR schedule 이 모두 train 'loss' 기준으로 동작합니다.")
    if args.data_seed is None and not args.val_annotation_file:
        print('[WARN] --data_seed 가 없어 실행마다 train/val split 이 달라집니다. '
              '체크포인트를 이어 학습할 계획이면 seed 를 고정하세요 (val 누수 방지).')

    # ----- multiscale / augment 설정 (간단 버전: 사용 안함) -----
    rescale_interval = -1       # multiscale X
    enhance_augment = None      # mosaic X
    multi_anchor_assign = False

    # ----- Epoch 계획 검증 -----
    initial_epoch = args.init_epoch
    transfer_epochs = args.transfer_epoch
    total_epochs = args.total_epoch

    if total_epochs <= initial_epoch:
        raise ValueError(
            f'--total_epoch({total_epochs}) 는 --init_epoch({initial_epoch}) 보다 '
            f'커야 합니다.')
    if transfer_epochs > 0 and total_epochs <= initial_epoch + transfer_epochs:
        raise ValueError(
            f'--total_epoch({total_epochs}) 가 --init_epoch({initial_epoch}) + '
            f'--transfer_epoch({transfer_epochs}) 이하라 fine-tune 단계가 '
            f'조용히 스킵됩니다. 값을 조정하세요.')

    # ----- Optimizer & Pruning 설정 -----
    steps_per_epoch = max(1, num_train // args.batch_size)
    validation_steps = max(1, num_val // args.batch_size) if has_val else 0
    # 실제로 돌게 될 step 수 기준으로 pruning schedule 종료 지점을 잡는다
    pruning_end_step = steps_per_epoch * (total_epochs - initial_epoch) \
        if args.model_pruning else 0

    # TF 2.11+ 는 Adam(lr=..., decay=...) 를 더 이상 받지 않는다
    optimizer = build_adam(learning_rate=args.learning_rate, decay=0.0)

    # ----- Tiny YOLOv3 Train Model 생성 -----
    model_type = 'tiny_yolo3_darknet'

    model, model_body = get_yolo3_train_model(
        model_type=model_type,
        anchors=anchors,
        num_classes=num_classes,
        weights_path=args.weights_path,
        freeze_level=args.freeze_level,
        optimizer=optimizer,
        label_smoothing=args.label_smoothing,
        elim_grid_sense=args.elim_grid_sense,
        model_pruning=args.model_pruning,
        pruning_end_step=pruning_end_step,
        # Vitis-AI quantize/compile 은 (None, None, 3) 입력을 받지 못한다.
        # 고정 shape 를 그래프에 박아 넣는다.
        input_shape=input_shape,
    )

    model.summary()

    # pretrained weights 로드에 실패하면 get_yolo3_train_model 이 freeze_level 을
    # 0 으로 낮춘다. 그러면 "frozen transfer stage" 는 아무것도 얼지 않은 채
    # 학습을 두 조각으로 나누기만 하므로 그냥 통짜로 학습한다.
    effective_freeze_level = getattr(model, 'effective_freeze_level',
                                     args.freeze_level)
    if transfer_epochs > 0 and effective_freeze_level == 0:
        print('[INFO] freeze 된 레이어가 없으므로 transfer stage 를 생략하고 '
              'full training 으로 진행합니다.')
        transfer_epochs = 0

    # ----- Data Generator 구성 -----
    train_gen = yolo3_data_generator_wrapper(
        train_dataset, args.batch_size, input_shape,
        anchors, num_classes, enhance_augment,
        rescale_interval, multi_anchor_assign=multi_anchor_assign,
        augment=True, shuffle=True
    )

    # 검증 데이터는 절대 증강하지 않는다. 증강하면 val_loss 가 노이즈가 되고
    # val_loss 기준 콜백들이 "운 좋게 쉬운 증강이 걸린 epoch" 을 고르게 된다.
    val_gen = yolo3_data_generator_wrapper(
        val_dataset, args.batch_size, input_shape,
        anchors, num_classes, None,
        -1, multi_anchor_assign=multi_anchor_assign,
        augment=False, shuffle=False
    ) if has_val else None

    # ----- 공용 콜백 -----
    epoch_tracker = EpochTracker(init_epoch=initial_epoch)
    best_metric_ckpt = BestMetricCheckpoint(
        log_dir=log_dir,
        monitor=monitor,   # 나중에 mAP metric 추가 시 'val_mAP' + mode='max'
        mode='min',
        filename_prefix='best',
    )
    checkpoint_clean = CheckpointCleanCallBack(
        log_dir, max_val_keep=5, max_eval_keep=2, max_best_keep=3
    )
    shared_callbacks = [epoch_tracker, best_metric_ckpt, checkpoint_clean]

    def sample_banner():
        print(
            'Train on {} samples, val on {} samples, batch size {}, '
            'input_shape {}.'.format(
                num_train, num_val, args.batch_size, input_shape
            )
        )

    # =====================================================
    # 4. Training Loop + KeyboardInterrupt 처리
    # =====================================================
    try:
        if transfer_epochs > 0:
            # ----- 1단계: freeze 상태 transfer training -----
            print("=== Transfer training (frozen) stage ===")
            sample_banner()

            fit_model(
                model, train_gen, steps_per_epoch, val_gen, validation_steps,
                epochs=initial_epoch + transfer_epochs,
                initial_epoch=initial_epoch,
                callbacks=make_stage_callbacks(
                    log_dir, monitor, shared_callbacks, args.model_pruning),
            )

            # ----- Freeze 풀고 전체 fine-tune -----
            print("=== Unfreeze and continue training, to fine-tune. ===")
            for layer in model.layers:
                layer.trainable = True
            model.compile(
                optimizer=optimizer,
                loss={'yolo_loss': lambda y_true, y_pred: y_pred}
            )
            sample_banner()

            fit_model(
                model, train_gen, steps_per_epoch, val_gen, validation_steps,
                epochs=total_epochs,
                initial_epoch=initial_epoch + transfer_epochs,
                callbacks=make_stage_callbacks(
                    log_dir, monitor, shared_callbacks, args.model_pruning),
            )
        else:
            # freeze 단계 없이 바로 full training
            print("=== Full training (no frozen stage) ===")
            sample_banner()

            fit_model(
                model, train_gen, steps_per_epoch, val_gen, validation_steps,
                epochs=total_epochs,
                initial_epoch=initial_epoch,
                callbacks=make_stage_callbacks(
                    log_dir, monitor, shared_callbacks, args.model_pruning),
            )

    except KeyboardInterrupt:
        print("\n[WARN] Training interrupted by user (Ctrl+C).")
        current_epoch = epoch_tracker.epoch
        print(f"[INFO] Last finished epoch: {current_epoch}")

        save_models(model, model_body, log_dir, current_epoch, args,
                    suffix='interrupt')
        report_best(best_metric_ckpt)
        return

    # =====================================================
    # 5. 정상 종료 시: 최종 모델 저장
    # =====================================================
    save_models(model, model_body, log_dir, epoch_tracker.epoch, args,
                suffix='final')
    report_best(best_metric_ckpt)


def save_models(model, model_body, log_dir, epoch, args, suffix='final'):
    """train model / weights / inference model 을 저장."""
    # --- 1) train model (pruning 제거 후) ---
    train_model = model
    if args.model_pruning:
        try:
            from tensorflow_model_optimization.sparsity import keras as sparsity
            train_model = sparsity.strip_pruning(model)
            print('[INFO] pruning wrapper stripped from train model.')
        except Exception as e:
            print(f'[WARN] strip_pruning on train model failed: {e}')

    train_path = os.path.join(log_dir, f'trained_{suffix}.h5')
    try:
        train_model.save(train_path)
        print(f"[INFO] Train model saved to: {train_path}")
        print("[NOTE] 이 파일은 yolo_loss Lambda 를 포함하므로 load_model() 시 "
              "custom_objects 가 필요합니다. 재개용으로는 아래 weights 파일을 쓰세요.")
    except Exception as e:
        print(f'[WARN] full train model save failed ({e}); weights 만 저장합니다.')

    # --- 2) weights-only (custom_objects 없이 항상 다시 읽을 수 있음) ---
    weights_path = os.path.join(log_dir, f'trained_{suffix}_weights.h5')
    try:
        train_model.save_weights(weights_path)
        print(f"[INFO] Train weights saved to: {weights_path}")
    except Exception as e:
        print(f'[WARN] weights save failed: {e}')

    # --- 3) inference model (Vitis-AI 로 넘길 산출물) ---
    # get_yolo3_train_model() 이 돌려준 model_body 를 그대로 쓴다.
    # (model.layers 안에서 중첩 Model 을 찾는 방식은 flat 모델이라 항상 실패한다)
    inf_model = build_inference_model(model_body, model_pruning=args.model_pruning)
    if inf_model is None:
        print("[ERROR] model_body is None; inference model not saved.")
        return

    inf_path = os.path.join(
        log_dir, f"inference_tiny_yolov3_epoch{epoch:03d}_{suffix}.h5")
    try:
        inf_model.save(inf_path)
        print(f"[INFO] Inference model saved to: {inf_path}")
        print(f"[INFO] inference input shape: {inf_model.input_shape}")
    except Exception as e:
        print(f'[WARN] inference model save failed: {e}')


def report_best(best_metric_ckpt):
    if best_metric_ckpt.best_path is not None:
        print(
            f"[INFO] Best checkpoint "
            f"({best_metric_ckpt.monitor}={best_metric_ckpt.best:.6f} "
            f"@ epoch {best_metric_ckpt.best_epoch}):"
        )
        print(f"       {best_metric_ckpt.best_path}")
    else:
        print("[INFO] No best checkpoint was saved (monitor metric may be missing).")


# =====================================================
# 6. Argument Parser
# =====================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 필수 경로 인자들
    parser.add_argument('--anchors_path', type=str, required=True,
                        help='path to tiny yolo3 anchor definitions')
    parser.add_argument('--classes_path', type=str, required=True,
                        help='path to class definitions')
    parser.add_argument('--annotation_file', type=str, required=True,
                        help='train annotation txt file (yolo format)')
    parser.add_argument('--val_annotation_file', type=str, default=None,
                        help='optional val annotation txt file')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='if no val_annotation_file, split ratio for validation')
    parser.add_argument('--data_seed', type=int, default=None,
                        help='seed for the train/val split shuffle. 고정하면 실행 간 '
                             'split 이 재현되어 재개 학습 시 val 누수를 막는다')

    parser.add_argument('--model_input_shape', type=str, default='256x256',
                        help="model input shape as <height>x<width>, multiples of 32")
    parser.add_argument('--weights_path', type=str, default=None,
                        help="pretrained weights h5 (by_name + skip_mismatch 로 로드). "
                             "로드에 실패하면 freeze_level 이 0 으로 낮아진다")

    # Training 설정
    parser.add_argument('--batch_size', type=int, default=16,
                        help="batch size for training")
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help="initial learning rate")
    parser.add_argument('--transfer_epoch', type=int, default=0,
                        help="frozen(backbone) training epochs. pretrained weights 가 "
                             "없으면 무시된다")
    parser.add_argument('--freeze_level', type=int, default=0, choices=[0, 1, 2],
                        help="0: no freeze, 1: freeze backbone, 2: freeze all but last layers")
    parser.add_argument('--init_epoch', type=int, default=0,
                        help="initial epoch index")
    parser.add_argument('--total_epoch', type=int, default=250,
                        help="total training epochs")
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help="label smoothing factor")
    parser.add_argument('--elim_grid_sense', action='store_true',
                        help="eliminate grid sensitivity")
    parser.add_argument('--model_pruning', action='store_true',
                        help="enable model pruning (tfmot magnitude pruning)")

    args = parser.parse_args()
    main(args)
