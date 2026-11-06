#!/bin/sh
# Quantize the best QAT checkpoint to the 8-bit checkpoint that the synthesis tool reads.
# Run from inside the ai8x-synthesis folder.
python quantize.py trained/qat_best.pth.tar trained/ecg-qat8-q.pth.tar --device MAX78002 -v "$@"
