#!/bin/sh
# =====================================================================================
# Task 2 pipeline: baselines -> (optional model selection) -> final model -> evaluation
#
#   sh run_task2.sh                  baselines + train final model + VALIDATION evaluation
#   SWEEP=1 sh run_task2.sh          first run the model-selection experiments (4.4), then train the winner
#   ONLY_FINAL=1 sh run_task2.sh     NO training: only the final DS2 evaluation, noise test and error analysis
#                                    (run this ONCE, after you are happy with the validation results)
#   FINAL=1 sh run_task2.sh          training AND the final steps in one go
#   QUICK=1 sh run_task2.sh          very short training, only to check that everything runs
#
# If configs/best_config.yaml exists (written by the sweep) it is used automatically.
# Run it from the project root. Needs data/processed (from src/prepare_data.py) and, for the noise
# test, data/raw and data/nstdb (downloaded by prepare_data.py).
# =====================================================================================
set -e
cd "$(dirname "$0")/src"
BASE=../configs/edge_cnn.yaml
CFG=$BASE
if [ -f ../configs/best_config.yaml ]; then CFG=../configs/best_config.yaml; fi
EXTRA=""
if [ "${QUICK:-0}" = "1" ]; then EXTRA="--set training.epochs=3 qat.epochs=2"; fi
echo "Using config: $CFG"

if [ "${ONLY_FINAL:-0}" != "1" ]; then
  echo "== 1. Classical baselines (validation only)";           python baseline.py --config $BASE
  echo "== 2. Deep-learning baseline: tiny CNN";                python train.py --config ../configs/tiny_cnn.yaml $( [ "${QUICK:-0}" = "1" ] && echo "--set training.epochs=3" )
  if [ "${SWEEP:-0}" = "1" ]; then
    echo "== 3. Model selection (validation macro-F1)";          python train.py --config $BASE --sweep ../configs/sweep.yaml
    CFG=../configs/best_config.yaml
  fi
  echo "== 4. Final model: FP32 training, then QAT (INT8)";     python train.py --config $CFG --name edge_cnn $EXTRA
  echo "== 5. Evaluation on the validation records";            python evaluate.py --config $CFG --split val
fi

if [ "${FINAL:-0}" = "1" ] || [ "${ONLY_FINAL:-0}" = "1" ]; then
  echo "== 6. FINAL evaluation on DS2 (clean)";                 python evaluate.py --config $CFG --split test --final
  echo "== 7. Noise robustness (12, 6, 0 dB) + filter ablation"; python noise_test.py --config $CFG --final --ablate-filter
  echo "== 8. Error analysis";                                  python error_analysis.py --config $CFG --final
fi
echo "== 9. Generate the MAX78002 files (see max78002/README_toolchain.md)"
python export_ai8x.py --config $CFG --out ../max78002
echo "Done. Results are in results/, models in models/."
