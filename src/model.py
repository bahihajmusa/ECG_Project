"""
model.py
========
The deep-learning models of the project (plain PyTorch).

EdgeECGNet is a small 1D CNN made ONLY of operations that the MAX78002 CNN
accelerator supports:

    [MaxPool1d] -> Conv1d (kernel 1..5, 'same' padding) -> BatchNorm -> ReLU     (x 6)
    flatten -> Linear (4 outputs)

The order "pool first, then convolution" copies how the Analog Devices
(ai8x) fused layers work (FusedMaxPoolConv1dBNReLU): the pooling is applied to
the INPUT of the convolution. Because of this, the same layer list can be used to
generate the ai8x-training model and the ai8x-synthesis network file
(see export_ai8x.py), so the model trained here and the model synthesized for the
chip have the same topology.

Software INT8 emulation (for the FP32 versus INT8 comparison)
------------------------------------------------------------
set_quant('ptq') or set_quant('qat') makes every layer behave like the 8-bit
hardware:
  * BatchNorm is folded into the convolution (hardware has no BN layer),
  * weights and biases are rounded to signed 8-bit values (power-of-two scale per layer),
  * the input and every activation are rounded to 8 bit with step 1/128 and clamped
    to [-1, 127/128] (the activation range of the MAX78002 CNN accelerator),
  * the last Linear layer keeps a wide (32-bit) output, like the 'wide' layer of ai8x.
'ptq'  = post-training quantization (no retraining).
'qat'  = quantization-aware training (fake quantization with a straight-through
         estimator; BN statistics are frozen, as in ai8x).
This is an EMULATION. It is not bit-exact with the real chip. The official numbers
come from the ai8x-training / ai8x-synthesis toolchain (see max78002/).
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Architecture presets (each dict = one fused layer)
#   out  : output channels
#   k    : kernel size
#   pool : max-pool size applied to the INPUT of this layer (0 = no pooling)
# ----------------------------------------------------------------------------
ARCHS: Dict[str, List[Dict]] = {
    # final proposed edge model (about 25k parameters)
    'edge_cnn': [
        {'out': 16, 'k': 5, 'pool': 0},
        {'out': 32, 'k': 5, 'pool': 2},
        {'out': 32, 'k': 5, 'pool': 2},
        {'out': 48, 'k': 3, 'pool': 2},
        {'out': 64, 'k': 3, 'pool': 2},
        {'out': 32, 'k': 1, 'pool': 2},
    ],
    # very small neural network (deep-learning baseline, about 3k parameters)
    'tiny_cnn': [
        {'out': 8, 'k': 5, 'pool': 0},
        {'out': 16, 'k': 5, 'pool': 4},
        {'out': 16, 'k': 3, 'pool': 4},
    ],
}

# Limits used by check_max78002_constraints (from the ai8x documentation / source):
MAX78002_MAX_WEIGHT_BYTES = 2 * 1024 * 1024      # 2 MB weight memory (project limit as well)
MAX78002_MAX_KERNEL_1D = 9
MAX78002_MAX_PAD_1D = 2
MAX78002_MAX_POOL = 16
MAX78002_MAX_LAYERS = 128


# ----------------------------------------------------------------------------
# INT8 helpers
# ----------------------------------------------------------------------------

def fake_quant(x: torch.Tensor) -> torch.Tensor:
    """
    Round to signed 8 bit with step 1/128 and clamp to [-1, 127/128].
    Rounding uses a straight-through estimator (gradient passes as if no rounding);
    values outside the range get zero gradient (clamp).
    """
    xc = torch.clamp(x * 128.0, -128.0, 127.0)
    q = xc + (torch.round(xc) - xc).detach()
    return q / 128.0


def pow2_scale(*tensors: torch.Tensor) -> float:
    """Smallest power of two >= the largest absolute value (the 'output shift' of the hardware)."""
    m = max(float(t.detach().abs().max()) for t in tensors)
    if m < 1e-8:
        return 1.0
    return float(2.0 ** math.ceil(math.log2(m)))


def quantize_weights(w: torch.Tensor, b: Optional[torch.Tensor]):
    """8-bit weights and bias with one power-of-two scale per layer."""
    s = pow2_scale(w, b) if b is not None else pow2_scale(w)
    wq = fake_quant(w / s) * s
    bq = fake_quant(b / s) * s if b is not None else None
    return wq, bq


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """[MaxPool1d] -> Conv1d -> BatchNorm1d -> ReLU, with optional INT8 emulation."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int = 0):
        super().__init__()
        self.pool_size = int(pool)
        self.kernel = int(kernel)
        self.pad = self.kernel // 2          # 'same' padding for odd kernels (k=1 -> 0, 3 -> 1, 5 -> 2)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=self.pad, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.quant = 'off'
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def train(self, mode: bool = True):
        super().train(mode)
        if self.quant != 'off':               # BN statistics are frozen while quantizing (as in ai8x)
            self.bn.eval()
        return self

    def folded_params(self):
        """Convolution weight and bias with the BatchNorm folded in (what the chip stores)."""
        std = torch.sqrt(self.bn.running_var + self.bn.eps)
        g = self.bn.weight / std
        w = self.conv.weight * g.view(-1, 1, 1)
        b = self.bn.bias - self.bn.running_mean * g
        return w, b

    def forward(self, x):
        if self.pool_size > 1:
            x = F.max_pool1d(x, self.pool_size, self.pool_size)
        if self.quant == 'off':
            return torch.clamp(F.relu(self.bn(self.conv(x))), max=127.0 / 128.0)
        w, b = self.folded_params()
        wq, bq = quantize_weights(w, b)
        y = F.relu(F.conv1d(x, wq, bq, padding=self.pad))
        return fake_quant(y)


class EdgeECGNet(nn.Module):
    """Small 1D CNN for 4-class heartbeat classification (input shape (batch, 1, 360))."""

    def __init__(self, layers: List[Dict], num_classes: int = 4, in_channels: int = 1,
                 input_length: int = 360, dropout: float = 0.0):
        super().__init__()
        self.layer_specs = [dict(s) for s in layers]
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.input_length = input_length
        blocks, ch, length = [], in_channels, input_length
        for s in self.layer_specs:
            blocks.append(ConvBlock(ch, s['out'], s['k'], s.get('pool', 0)))
            if s.get('pool', 0) > 1:
                length = length // s['pool']
            ch = s['out']
        self.blocks = nn.ModuleList(blocks)
        self.flat_features = ch * length
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.flat_features, num_classes, bias=True)
        self.quant = 'off'

    def set_quant(self, mode: str):
        """mode: 'off' (FP32), 'ptq' or 'qat' (INT8 emulation)."""
        assert mode in ('off', 'ptq', 'qat')
        self.quant = mode
        for b in self.blocks:
            b.quant = mode
        self.train(self.training)             # re-apply the BN freezing rule
        return self

    def forward(self, x):
        if self.quant != 'off':
            x = fake_quant(x)                 # the chip receives 8-bit input
        for b in self.blocks:
            x = b(x)
        x = x.flatten(1)
        x = self.dropout(x)
        if self.quant == 'off':
            return self.fc(x)
        wq, bq = quantize_weights(self.fc.weight, self.fc.bias)
        return F.linear(x, wq, bq)            # wide (32-bit) output: no activation clamp


# ----------------------------------------------------------------------------
# Construction from a config
# ----------------------------------------------------------------------------

def _round4(c: float) -> int:
    return max(4, int(round(c / 4.0)) * 4)


def make_layer_specs(arch: str = 'edge_cnn', width_mult: float = 1.0,
                     layers: Optional[List[Dict]] = None) -> List[Dict]:
    """Layer list of a preset (or an explicit list), with channels scaled by width_mult."""
    base = [dict(s) for s in (layers if layers is not None else ARCHS[arch])]
    if abs(width_mult - 1.0) > 1e-9:
        for s in base:
            s['out'] = _round4(s['out'] * width_mult)
    return base


def build_model(model_cfg: Dict) -> EdgeECGNet:
    """model_cfg keys: arch, width_mult, layers (optional), dropout, num_classes, input_length."""
    specs = make_layer_specs(model_cfg.get('arch', 'edge_cnn'), model_cfg.get('width_mult', 1.0),
                             model_cfg.get('layers'))
    return EdgeECGNet(specs, num_classes=model_cfg.get('num_classes', 4),
                      in_channels=model_cfg.get('in_channels', 1),
                      input_length=model_cfg.get('input_length', 360),
                      dropout=model_cfg.get('dropout', 0.0))


# ----------------------------------------------------------------------------
# Description tables (report 4.2 and 6.1) and MAX78002 checks
# ----------------------------------------------------------------------------

def count_parameters(model: EdgeECGNet, deployed: bool = False) -> int:
    """
    deployed=False : all trainable parameters (conv weights + BN gamma/beta + FC).
    deployed=True  : parameters stored on the chip: conv weights + folded bias + FC weights + bias
                     (BatchNorm is folded away, so BN adds nothing).
    """
    if not deployed:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    n = 0
    for b in model.blocks:
        n += b.conv.weight.numel() + b.conv.out_channels      # weights + folded bias
    n += model.fc.weight.numel() + model.fc.bias.numel()
    return n


def describe_architecture(model: EdgeECGNet) -> List[Dict]:
    """One row per layer: Layer, Input, Operation, Output, Parameters (deployed, BN folded)."""
    rows, ch, length = [], model.in_channels, model.input_length
    for i, b in enumerate(model.blocks, 1):
        op = []
        in_len = length
        if b.pool_size > 1:
            op.append(f'MaxPool1d({b.pool_size})')
            length = length // b.pool_size
        op.append(f'Conv1d(k={b.kernel}, pad={b.pad}) + BN(folded) + ReLU')
        out_ch = b.conv.out_channels
        params = b.conv.weight.numel() + out_ch
        rows.append({'Layer': f'conv{i}', 'Input': f'{ch} x {in_len}',
                     'Operation': ' -> '.join(op), 'Output': f'{out_ch} x {length}', 'Parameters': params})
        ch = out_ch
    rows.append({'Layer': 'fc', 'Input': f'{ch} x {length} = {ch * length}',
                 'Operation': 'Flatten -> Linear (32-bit output)', 'Output': f'{model.num_classes}',
                 'Parameters': model.fc.weight.numel() + model.fc.bias.numel()})
    return rows


def architecture_markdown(model: EdgeECGNet) -> str:
    rows = describe_architecture(model)
    lines = ['| Layer | Input (C x L) | Operation | Output | Parameters |', '|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['Layer']} | {r['Input']} | {r['Operation']} | {r['Output']} | {r['Parameters']:,} |")
    lines.append(f"| **Total (deployed)** | | | | **{count_parameters(model, deployed=True):,}** |")
    return '\n'.join(lines)


def check_max78002_constraints(model: EdgeECGNet) -> Dict:
    """
    Check the model against the MAX78002 rules we rely on. Returns a dict with
    'ok', a list of 'problems', and a few numbers (parameters, weight bytes, layer count).
    The real decision is made by the ai8x toolchain (see max78002/); this is an early warning.
    """
    problems = []
    if len(model.blocks) + 1 > MAX78002_MAX_LAYERS:
        problems.append(f'too many layers ({len(model.blocks) + 1} > {MAX78002_MAX_LAYERS})')
    for i, b in enumerate(model.blocks, 1):
        if b.kernel > MAX78002_MAX_KERNEL_1D:
            problems.append(f'conv{i}: kernel {b.kernel} > {MAX78002_MAX_KERNEL_1D}')
        if b.pad > MAX78002_MAX_PAD_1D:
            problems.append(f'conv{i}: padding {b.pad} > {MAX78002_MAX_PAD_1D}')
        if b.pool_size > MAX78002_MAX_POOL:
            problems.append(f'conv{i}: pool size {b.pool_size} > {MAX78002_MAX_POOL}')
    n = count_parameters(model, deployed=True)
    if n > MAX78002_MAX_WEIGHT_BYTES:         # 8-bit weights: 1 parameter = 1 byte
        problems.append(f'{n} bytes of 8-bit weights exceed the 2 MB limit')
    return {'ok': not problems, 'problems': problems, 'parameters_deployed': n,
            'weight_bytes_int8': n, 'layers': len(model.blocks) + 1}


# ----------------------------------------------------------------------------
# Checkpoint helpers
# ----------------------------------------------------------------------------

def save_checkpoint(path: str, model: EdgeECGNet, model_cfg: Dict, extra: Optional[Dict] = None):
    """Save weights together with everything needed to rebuild the model."""
    import os
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    payload = {'state_dict': model.state_dict(), 'model_cfg': model_cfg,
               'layer_specs': model.layer_specs, 'quant_mode': model.quant}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_model(path: str, device='cpu', quant: Optional[str] = None) -> EdgeECGNet:
    """
    Rebuild a model from a checkpoint made by save_checkpoint.
    quant=None keeps the mode stored in the checkpoint ('off' for FP32, 'qat' for the QAT model);
    pass 'off', 'ptq' or 'qat' to override (for example 'ptq' on the FP32 checkpoint).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model({**ckpt['model_cfg'], 'layers': ckpt['layer_specs'], 'width_mult': 1.0})
    model.load_state_dict(ckpt['state_dict'])
    model.set_quant(quant if quant is not None else ckpt.get('quant_mode', 'off'))
    return model.to(device).eval()


if __name__ == '__main__':
    m = build_model({'arch': 'edge_cnn'})
    x = torch.randn(4, 1, 360)
    print(architecture_markdown(m))
    print('FP32 out:', m(x).shape, '| trainable params:', count_parameters(m), '| deployed:', count_parameters(m, True))
    m.set_quant('ptq')
    print('INT8-emulated out:', m(x).shape)
    print(check_max78002_constraints(m))
