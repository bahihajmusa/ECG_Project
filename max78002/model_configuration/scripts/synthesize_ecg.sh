#!/bin/sh
# Generate the MAX78002 C project (weights, configuration, test code) with ai8x-synthesis.
# Run from inside the ai8x-synthesis folder.
python ai8xize.py --verbose --test-dir synthesis_output --prefix ecg_mitbih \
  --checkpoint-file trained/ecg-qat8-q.pth.tar --config-file networks/ecg_mitbih.yaml \
  --sample-input tests/sample_ecg_mitbih.npy --device MAX78002 \
  --compact-data --mexpress --timer 0 --display-checkpoint --overwrite --softmax "$@"
