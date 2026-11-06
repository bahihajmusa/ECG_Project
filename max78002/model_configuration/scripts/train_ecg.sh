#!/bin/sh
# Train (FP32 until the QAT start epoch, then quantization-aware) with ai8x-training.
# Run this from inside the ai8x-training folder (after copying the files, see README_toolchain.md).
python train.py --epochs 40 --optimizer Adam --lr 0.001 --wd 0 --deterministic \
  --compress policies/schedule_ecg.yaml --qat-policy policies/qat_policy_ecg.yaml \
  --model ai85ecgnet --dataset ECG_MITBIH --use-bias --batch-size 128 \
  --confusion --param-hist --device MAX78002 "$@"
