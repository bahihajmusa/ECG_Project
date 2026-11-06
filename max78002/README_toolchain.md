# MAX78002 deployment feasibility (report section 6)

This folder shows that the proposed model can be trained, quantized to 8 bit and synthesized for the
Analog Devices **MAX78002** with the official software toolchain. No board is needed.

```
max78002/
  model_configuration/   files for the toolchain, GENERATED from the model definition (src/export_ai8x.py)
      models/ai85net-ecg.py        ai8x-training model (same layers as src/model.py)
      datasets/ecg_mitbih.py       ai8x-training dataset (our prepared beats, validation split by record)
      networks/ecg_mitbih.yaml     ai8x-synthesis network description (layer by layer)
      policies/                    learning-rate schedule and QAT policy
      scripts/                     train / quantize / evaluate / synthesize commands
      sample_input.npy             one validation beat as 8-bit input
      model_summary.json           layers, parameters, estimated memory
  quantization_output/   (created by run_toolchain.sh) quantized checkpoint + logs
  synthesis_output/      (created by run_toolchain.sh) generated C project + synthesis_summary.md
  run_toolchain.sh       runs the whole flow
  collect_evidence.py    turns the tool output into synthesis_summary.md/.json for the report
```

## 1. One-time setup
The toolchain is separate software from Analog Devices. Use Python 3.11 in a separate environment:

```bash
git clone --recursive https://github.com/analogdevicesinc/ai8x-training.git
git clone --recursive https://github.com/analogdevicesinc/ai8x-synthesis.git
# follow the "Installation" chapter of each README (pip install -r requirements.txt in a virtual environment)
```
Both repositories must work before you continue. Read the version notes in their README files; this project was
checked against the `develop` branches.

## 2. Run everything
```bash
# from the project root
AI8X_TRAINING=/path/to/ai8x-training AI8X_SYNTHESIS=/path/to/ai8x-synthesis sh max78002/run_toolchain.sh
```
Steps: generate files, copy them into the two repositories, train (FP32 then QAT, batch-norm folded, 8-bit weights and
activations), quantize, evaluate the 8-bit model, synthesize the C project, collect the evidence. Training uses the
**validation** records as "test" set on purpose: the final test set DS2 is only copied/used when you set `FINAL=1`
(do that once, at the very end).

## 3. What goes into the report (section 6)
* 6.1 Model architecture: `model_configuration/model_summary.json` (and `src/model.py`).
* 6.2 Quantization: `quantization_output/` (quantize log, 8-bit checkpoint, ai8x evaluation logs) and `results/tables_*.md` (FP32 versus INT8).
* 6.3 MAX78002 compatibility: `synthesis_output/synthesis_summary.md` (PASSED/FAILED, weight and bias memory, operations per layer).
* 6.4 Limitations: no board measurements (latency, energy, real RAM), see the README of the project root.

## 4. Notes
* `weight memory` and `bias memory` come from the synthesis tool itself. The data (activation) memory is not printed by the tool;
  `model_summary.json` contains a simple estimate (largest input + output of one layer in bytes).
* The software INT8 emulation in `src/model.py` is not bit-exact with the chip. The official 8-bit numbers are the ones from
  `quantization_output/ai8x_evaluation_*.log`.
* If a step fails, the synthesis result is reported as FAILED with the problem listed; fix the model (for example kernel
  size, pooling or channel limits) in `configs/edge_cnn.yaml` / `src/model.py`, run `export_ai8x.py` again and repeat.
