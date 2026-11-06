#!/bin/sh
# =====================================================================================
# End-to-end MAX78002 flow: ai8x-training (QAT) -> quantize -> evaluate -> ai8x-synthesis
#
#   AI8X_TRAINING=/path/to/ai8x-training  AI8X_SYNTHESIS=/path/to/ai8x-synthesis  \
#       sh max78002/run_toolchain.sh
#
# Run it from the project root (the folder that contains src/, data/, max78002/).
# Prerequisites: README_toolchain.md section 1 (clone both repositories and install their
# requirements). The validation records are used for ai8x training/evaluation; the final test
# set DS2 is NOT copied unless you set FINAL=1 (do that only at the very end).
# =====================================================================================
set -e

: "${AI8X_TRAINING:?Set AI8X_TRAINING to your ai8x-training folder}"
: "${AI8X_SYNTHESIS:?Set AI8X_SYNTHESIS to your ai8x-synthesis folder}"
ROOT=$(pwd)
MC="$ROOT/max78002/model_configuration"
OUT_Q="$ROOT/max78002/quantization_output"
OUT_S="$ROOT/max78002/synthesis_output"
EPOCHS=${EPOCHS:-100}
QAT_START=${QAT_START:-60}
mkdir -p "$OUT_Q" "$OUT_S"


echo "== 2. Copy files into ai8x-training and ai8x-synthesis"
cp "$MC/models/ai85net-ecg.py"        "$AI8X_TRAINING/models/"
cp "$MC/datasets/ecg_mitbih.py"       "$AI8X_TRAINING/datasets/"
cp "$MC/policies/schedule_ecg.yaml"   "$AI8X_TRAINING/policies/"
cp "$MC/policies/qat_policy_ecg.yaml" "$AI8X_TRAINING/policies/"
mkdir -p "$AI8X_TRAINING/data/ECG_MITBIH"
for f in X_train y_train X_val y_val; do cp "data/processed/$f.npy" "$AI8X_TRAINING/data/ECG_MITBIH/"; done
if [ "${FINAL:-0}" = "1" ]; then
  for f in X_test y_test; do cp "data/processed/$f.npy" "$AI8X_TRAINING/data/ECG_MITBIH/"; done
fi
cp "$MC/networks/ecg_mitbih.yaml" "$AI8X_SYNTHESIS/networks/"
mkdir -p "$AI8X_SYNTHESIS/tests" "$AI8X_SYNTHESIS/trained"
cp "$MC/sample_input.npy" "$AI8X_SYNTHESIS/tests/sample_ecg_mitbih.npy"

echo "== 3. Train with ai8x-training (FP32, then quantization-aware)"
( cd "$AI8X_TRAINING" && sh "$MC/scripts/train_ecg.sh" ) 2>&1 | tee "$OUT_Q/train_console.log"
LOGDIR=$(ls -dt "$AI8X_TRAINING"/logs/* | head -1)
cp "$LOGDIR/qat_best.pth.tar" "$AI8X_SYNTHESIS/trained/qat_best.pth.tar"

echo "== 4. Quantize to 8 bit"
( cd "$AI8X_SYNTHESIS" && sh "$MC/scripts/quantize_ecg.sh" ) 2>&1 | tee "$OUT_Q/quantize_console.log"
cp "$AI8X_SYNTHESIS/trained/ecg-qat8-q.pth.tar" "$OUT_Q/"
cp "$AI8X_SYNTHESIS/trained/qat_best.pth.tar"   "$OUT_Q/"

echo "== 5. Evaluate the 8-bit model with ai8x-training (validation records)"
( cd "$AI8X_TRAINING" && CKPT="$AI8X_SYNTHESIS/trained/ecg-qat8-q.pth.tar" sh "$MC/scripts/evaluate_ecg.sh" ) 2>&1 | tee "$OUT_Q/ai8x_evaluation_validation.log"
if [ "${FINAL:-0}" = "1" ]; then
  echo "== 5b. FINAL evaluation on DS2 (use only once)"
  ( cd "$AI8X_TRAINING" && ECG_TEST_SET=ds2 CKPT="$AI8X_SYNTHESIS/trained/ecg-qat8-q.pth.tar" sh "$MC/scripts/evaluate_ecg.sh" ) 2>&1 | tee "$OUT_Q/ai8x_evaluation_test_DS2.log"
fi

echo "== 6. Synthesize the C project for MAX78002"
( cd "$AI8X_SYNTHESIS" && sh "$MC/scripts/synthesize_ecg.sh" --no-version-check ) 2>&1 | tee "$OUT_S/ai8xize_console.log"
rm -rf "$OUT_S/ecg_mitbih"
cp -r "$AI8X_SYNTHESIS/synthesis_output/ecg_mitbih" "$OUT_S/"

echo "== 7. Collect the evidence for the report"
python max78002/collect_evidence.py
echo "Done. See max78002/synthesis_output/synthesis_summary.md"
