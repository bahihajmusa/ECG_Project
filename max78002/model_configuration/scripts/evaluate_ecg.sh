#!/bin/sh
# Evaluate the quantized checkpoint with the ai8x-training framework (8-bit mode).
# Run from inside the ai8x-training folder. By default the "test" set is the VALIDATION records.
# For the FINAL evaluation on DS2 (once, at the very end):   ECG_TEST_SET=ds2 sh evaluate_ecg.sh
CKPT=${CKPT:-../ai8x-synthesis/trained/ecg-qat8-q.pth.tar}
python train.py --model ai85ecgnet --dataset ECG_MITBIH --use-bias --confusion --evaluate \
  --exp-load-weights-from "$CKPT" -8 --device MAX78002 "$@"
