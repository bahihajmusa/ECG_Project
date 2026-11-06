Filled by max78002/run_toolchain.sh:
  quantize_console.log           output of quantize.py (per-tensor ranges and 8-bit factors)
  qat_best.pth.tar               best quantization-aware-training checkpoint
  ecg-qat8-q.pth.tar             8-bit checkpoint used for synthesis
  train_console.log              ai8x-training console output
  ai8x_evaluation_validation.log accuracy/confusion matrix of the 8-bit model on the validation records
  ai8x_evaluation_test_DS2.log   (only with FINAL=1) final evaluation on DS2
