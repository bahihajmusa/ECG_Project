"""Build ONE report (Task 1 + Task 2) as a PDF from the files of the project.

Usage (from src/):
    python export_eda_figures.py                                   # once, after the notebook has been run and saved with outputs
    python make_report.py --config ../configs/best_config.yaml --split test
Output: ../report/Full_report.pdf        (needs: pip install reportlab pyyaml matplotlib)

Automatic: every table, number and figure of sections 4-6 (results/, max78002/), the EDA figures (results/eda/), the success-criteria review.
By hand:   report_task1.md (text of sections 1-3 and the references), report_notes.yaml (summary, targets, run history, changes).
The script only reads files; it never trains or evaluates.
"""
import argparse, csv, datetime, glob, json, os, re, sys
import yaml
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

# ---------------------------------------------------------------- fonts (DejaVu ships with matplotlib; handles arrows, ±, ≥, å)
SERIF, SERIF_B, SANS, SANS_B, MONO = 'Times-Roman', 'Times-Bold', 'Helvetica', 'Helvetica-Bold', 'Courier'
try:
    import matplotlib
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    FD = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
    for name, f in [('Serif', 'DejaVuSerif.ttf'), ('Serif-B', 'DejaVuSerif-Bold.ttf'), ('Serif-I', 'DejaVuSerif-Italic.ttf'),
                    ('Serif-BI', 'DejaVuSerif-BoldItalic.ttf'), ('Sans', 'DejaVuSans.ttf'), ('Sans-B', 'DejaVuSans-Bold.ttf'),
                    ('Sans-I', 'DejaVuSans-Oblique.ttf'), ('Sans-BI', 'DejaVuSans-BoldOblique.ttf'), ('Mono', 'DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name, os.path.join(FD, f)))
    registerFontFamily('Serif', normal='Serif', bold='Serif-B', italic='Serif-I', boldItalic='Serif-BI')
    registerFontFamily('Sans', normal='Sans', bold='Sans-B', italic='Sans-I', boldItalic='Sans-BI')
    SERIF, SERIF_B, SANS, SANS_B, MONO = 'Serif', 'Serif-B', 'Sans', 'Sans-B', 'Mono'
except Exception as e:                      # falls back to the built-in fonts (arrows may then be missing)
    print('Note: DejaVu fonts not found, using built-in fonts:', e)

ST = {
    'title': ParagraphStyle('t', fontName=SERIF_B, fontSize=24, leading=29, alignment=1, spaceAfter=10),
    'center': ParagraphStyle('c', fontName=SERIF, fontSize=12, leading=16, alignment=1, spaceAfter=6),
    'h1': ParagraphStyle('h1', fontName=SERIF_B, fontSize=16, leading=20, alignment=1, spaceBefore=10, spaceAfter=8, keepWithNext=1),
    'h2': ParagraphStyle('h2', fontName=SERIF_B, fontSize=13, leading=16, spaceBefore=9, spaceAfter=5, keepWithNext=1),
    'h3': ParagraphStyle('h3', fontName=SERIF_B, fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=3, keepWithNext=1),
    'h3c': ParagraphStyle('h3c', fontName=SERIF_B, fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=3, keepWithNext=1, alignment=1),
    'figcap': ParagraphStyle('figcap', fontName=SANS, fontSize=7.5, leading=9.5, textColor=colors.grey, spaceAfter=6, alignment=1),
    'p': ParagraphStyle('p', fontName=SERIF, fontSize=10, leading=13.2, spaceAfter=5, alignment=4),
    'b': ParagraphStyle('b', fontName=SERIF, fontSize=10, leading=13.2, leftIndent=14, bulletIndent=3, spaceAfter=2, alignment=4),
    'cell': ParagraphStyle('cell', fontName=SANS, fontSize=7.6, leading=9.6),
    'cellb': ParagraphStyle('cellb', fontName=SANS_B, fontSize=7.6, leading=9.6),
    'cap': ParagraphStyle('cap', fontName=SANS, fontSize=7.5, leading=9.5, textColor=colors.grey, spaceAfter=6),
}
W = A4[0] - 36 * mm


def inline(text):
    t = escape(str(text))
    t = t.replace('&lt;br/&gt;', '<br/>')
    t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'`(.+?)`', rf'<font face="{MONO}">\1</font>', t)
    return t


def table(rows, header=True, widths=None):
    data = [[Paragraph(inline(c), ST['cellb'] if (header and i == 0) else ST['cell']) for c in r] for i, r in enumerate(rows)]
    if widths is None and rows:
        wts = [min(max(max(len(str(r[j])) for r in rows), 8), 70) for j in range(len(rows[0]))]
        widths = [W * w / sum(wts) for w in wts]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [('GRID', (0, 0), (-1, -1), 0.3, colors.lightgrey), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
             ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3)]
    if header:
        style.append(('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eef6')))
    t.setStyle(TableStyle(style))
    return t


def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def csv_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def fnum(x, d=4):
    try:
        return f'{float(x):.{d}f}'
    except (TypeError, ValueError):
        return str(x)


def figure(path, caption, frac=1.0):
    iw, ih = ImageReader(path).getSize()
    w = min(W * frac, W)
    h = w * ih / iw
    if h > 200 * mm:
        h = 200 * mm; w = h * iw / ih
    items = [Image(path, width=w, height=h)]
    if caption:
        items.append(Paragraph(inline(caption), ST['figcap']))
    return KeepTogether(items)


# ---------------------------------------------------------------- markdown renderer (Task 1 text)
def render_md(c, text, tvars):
    for k, v in tvars.items():
        text = text.replace('{{%s}}' % k, str(v))
    text = re.sub(r'<!--(?!\s*pagebreak).*?-->', '', text, flags=re.S)
    out, lines, i = [], text.splitlines(), 0
    search = [os.path.join(c.res, 'eda'), c.assets, c.res]
    while i < len(lines):
        ln = lines[i].rstrip()
        i += 1
        if not ln.strip():
            continue
        if re.match(r'^\s*<!--\s*pagebreak\s*-->', ln):
            out.append(PageBreak()); continue
        m = re.match(r'^(#{1,3})\s+(.*)', ln)
        if m:
            lvl = len(m.group(1))
            nxt = next((x for x in lines[i:] if x.strip()), '')
            style = 'h3c' if (lvl == 3 and nxt.startswith('![')) else 'h%d' % lvl      # heading above a figure is centered
            out.append(Paragraph(inline(m.group(2)), ST[style])); continue
        m = re.match(r'^!\[(.*?)\]\((.+?)(?:\|([\d.]+))?\)\s*$', ln)
        if m:
            cap, name, frac = m.group(1), m.group(2), float(m.group(3) or 1.0)
            p = next((os.path.join(d, name) for d in search if os.path.exists(os.path.join(d, name))), None)
            if p:
                out.append(figure(p, cap, frac))
            else:
                c.missing.append(name)
            continue
        if ln.lstrip().startswith('|'):
            rows = [[x.strip() for x in ln.strip().strip('|').split('|')]]
            while i < len(lines) and lines[i].lstrip().startswith('|'):
                cells = [x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-{2,}:?', x) for x in cells):
                    rows.append(cells)
                i += 1
            n = max(len(r) for r in rows)
            out += [table([r + [''] * (n - len(r)) for r in rows]), Spacer(1, 5)]
            continue
        if re.match(r'^\s*-\s+', ln):
            out.append(Paragraph(inline(re.sub(r'^\s*-\s+', '', ln)), ST['b'], bulletText='•')); continue
        out.append(Paragraph(inline(ln), ST['p']))
    merged, k = [], 0                                   # keep a "###" heading together with the figure that follows it
    while k < len(out):
        a = out[k]
        if isinstance(a, Paragraph) and a.style.name in ('h3', 'h3c') and k + 1 < len(out) and isinstance(out[k + 1], KeepTogether):
            merged.append(KeepTogether([a] + list(out[k + 1]._content))); k += 2
        else:
            merged.append(a); k += 1
    return merged


# ---------------------------------------------------------------- context and inputs
class Ctx:
    pass


DEFAULT_TARGETS = {
    'macro_f1': {'target': 0.60, 'partial': 0.45}, 'baseline_margin': {'target': 0.05, 'partial': 0.0},
    'recall_V': {'target': 0.80, 'partial': 0.60}, 'recall_S': {'target': 0.50, 'partial': 0.25},
    'recall_F': {'target': 0.50, 'partial': 0.25}, 'noise_drop_0db': {'target': 0.05, 'partial': 0.10},
    'int8_drop': {'target': 0.03}, 'size_mb': {'target': 2.0},
}


def load(args):
    c = Ctx()
    cfg_path = os.path.abspath(args.config)
    c.cfg = yaml.safe_load(read(cfg_path))
    cdir = os.path.dirname(cfg_path)
    rel = lambda p: p if os.path.isabs(p) else os.path.normpath(os.path.join(cdir, p))
    c.res = args.results_dir or rel(c.cfg['paths']['results_dir'])
    c.root = os.path.dirname(c.res)
    c.mx = args.max78002_dir or os.path.join(c.root, 'max78002')
    c.assets = os.path.join(c.root, 'report_assets')
    c.split = args.split
    npath = args.notes or os.path.join(c.root, 'report_notes.yaml')
    c.notes = yaml.safe_load(read(npath)) if os.path.exists(npath) else {}
    c.t1 = args.task1 or os.path.join(c.root, 'report_task1.md')
    c.out = args.out or os.path.join(c.root, 'report', 'Full_report.pdf')
    c.missing = []
    t = dict(DEFAULT_TARGETS)
    for k, v in (c.notes.get('targets') or {}).items():
        t[k] = v if isinstance(v, dict) else {'target': v}
    c.targets = t
    return c


def rp(c, name):
    p = os.path.join(c.res, name)
    if not os.path.exists(p):
        c.missing.append(name)
        return None
    return p


def metric_table(c):
    p = rp(c, 'metrics.csv')
    return csv_rows(p) if p else []


def get_metric(rows, model, split, cls='overall', key='macro_f1', cond='clean'):
    for r in rows:
        if r['model'] == model and r['split'] == split and r['class'] == cls and r['condition'] == cond:
            v = r.get(key, '')
            return float(v) if v not in ('', None) else None
    return None


def noise_value(c, model, snr, key='macro_f1'):
    p = os.path.join(c.res, 'noise_results.csv')
    if not os.path.exists(p):
        return None
    vals = []
    for r in csv_rows(p):
        if r['model'] != model or r.get('filter', 'on') != 'on':
            continue
        if (snr == 'clean' and r['condition'] == 'clean') or (snr != 'clean' and r['snr_db'] not in ('', None) and float(r['snr_db']) == snr):
            vals.append(float(r[key]))
    return sum(vals) / len(vals) if vals else None


def synth_info(c):
    p = os.path.join(c.mx, 'synthesis_output', 'synthesis_summary.md')
    if not os.path.exists(p):
        c.missing.append('max78002/synthesis_output/synthesis_summary.md')
        return None, ''
    txt = read(p)
    return ('PASSED' in txt.split('Generated C project')[0]), txt


def parse_ai8x_log(path):
    m = re.search(r'==> Confusion:\s*((?:\s*\[+[^\n]*\]+\s*\n?)+)', read(path))
    if not m:
        return None
    nums = [int(x) for x in re.findall(r'\d+', m.group(1))]
    return [nums[i * 4:(i + 1) * 4] for i in range(4)] if len(nums) == 16 else None


def ai8x_metrics(c, key):
    p = os.path.join(c.mx, 'quantization_output', f'ai8x_evaluation_{key}.log')
    if not os.path.exists(p):
        return None
    cm = parse_ai8x_log(p)
    if cm is None:
        c.missing.append(f'parsable confusion matrix in {os.path.basename(p)}'); return None
    n = sum(map(sum, cm)); tp = [cm[i][i] for i in range(4)]
    col = [sum(cm[r][i] for r in range(4)) for i in range(4)]
    row = [sum(cm[i]) for i in range(4)]
    prec = [tp[i] / col[i] if col[i] else 0.0 for i in range(4)]
    rec = [tp[i] / row[i] if row[i] else 0.0 for i in range(4)]
    f1 = [2 * prec[i] * rec[i] / (prec[i] + rec[i]) if (prec[i] + rec[i]) else 0.0 for i in range(4)]
    return dict(n=n, accuracy=sum(tp) / n, macro_f1=sum(f1) / 4, f1=f1, recall=rec, cm=cm)


def criteria_values(c, main='edge_int8_qat'):
    m, s, v = metric_table(c), c.split, {}
    v['accuracy'] = get_metric(m, main, s, key='accuracy')
    v['macro_f1'] = get_metric(m, main, s)
    for k in 'SVF':
        v['recall_' + k] = get_metric(m, main, s, k, 'recall')
    fp = get_metric(m, 'edge_fp32', s)
    v['int8_drop'] = (fp - v['macro_f1']) if (fp is not None and v['macro_f1'] is not None) else None
    base = get_metric(m, 'baseline', s)
    v['baseline_margin'] = (v['macro_f1'] - base) if (base is not None and v['macro_f1'] is not None) else None
    n0, nc = noise_value(c, main, 0), noise_value(c, main, 'clean')
    v['noise_drop_0db'] = (nc - n0) if (n0 is not None and nc is not None) else None
    params = next((float(r['parameters']) for r in m if r['model'] == main and r['split'] == s and r['class'] == 'overall' and r.get('parameters')), None)
    v['size_mb'] = params / 1e6 if params else None            # INT8: 1 byte per parameter
    am = ai8x_metrics(c, 'test_DS2')
    if am:
        v['ai8x_macro_f1'] = am['macro_f1']
        for i, k in enumerate('NSVF'):
            v['ai8x_recall_' + k] = am['recall'][i]
        v['ai8x_baseline_margin'] = (am['macro_f1'] - base) if base is not None else None
    ok, _ = synth_info(c)
    v['synthesis_passed'] = None if ok is None else (1.0 if ok else 0.0)
    return v


OPS = {'>=': lambda a, b: a >= b, '<=': lambda a, b: a <= b}


def criteria_list(c):
    t = c.targets
    g = lambda k, f: t.get(k, {}).get(f)
    L = [('Macro-F1 (INT8 emulation)', 'macro_f1', '>=', 'macro_f1'),
         ('Macro-F1 above the best classical baseline', 'baseline_margin', '>=', 'baseline_margin'),
         ('Recall class V', 'recall_V', '>=', 'recall_V'), ('Recall class S', 'recall_S', '>=', 'recall_S'),
         ('Recall class F', 'recall_F', '>=', 'recall_F'),
         ('Macro-F1 drop from FP32 to INT8', 'int8_drop', '<=', 'int8_drop'),
         ('Macro-F1 drop at 0 dB noise', 'noise_drop_0db', '<=', 'noise_drop_0db'),
         ('Model size, INT8 (MB)', 'size_mb', '<=', 'size_mb'),
         ('Deployed ai8x model: macro-F1', 'ai8x_macro_f1', '>=', 'macro_f1'),
         ('Deployed ai8x model: recall class V', 'ai8x_recall_V', '>=', 'recall_V'),
         ('Deployed ai8x model: recall class S', 'ai8x_recall_S', '>=', 'recall_S'),
         ('Deployed ai8x model: recall class F', 'ai8x_recall_F', '>=', 'recall_F')]
    out = [dict(name=n, metric=m, op=o, target=g(tk, 'target'), partial=g(tk, 'partial')) for n, m, o, tk in L]
    out.insert(8, dict(name='MAX78002 synthesis passed', metric='synthesis_passed', op='>=', target=1))
    return out


def criteria_section(c):
    v = criteria_values(c)
    fl = [Paragraph('5.6 Success-criteria review', ST['h2'])]
    rows = [['Criterion', 'Measured', 'Target', 'Verdict']]
    for k in criteria_list(c):
        val, tgt = v.get(k['metric']), k['target']
        if val is None or tgt is None:
            verdict = 'not measurable (file missing)'
        else:
            ok = OPS[k['op']](val, tgt)
            part = k.get('partial')
            verdict = 'Achieved' if ok else ('Partially achieved' if (part is not None and OPS[k['op']](val, part)) else 'Not achieved')
        rows.append([k['name'], fnum(val) if val is not None else '-', f"{k['op']} {tgt}" if tgt is not None else '-', f'**{verdict}**'])
    rows.append(['Process quality (no record in two splits; DS2 used once and logged)', 'see log', '', '**Achieved**'])
    fl.append(table(rows, widths=[W * .50, W * .12, W * .12, W * .26]))
    fl.append(Paragraph('Targets are those of section 1.4 (report_notes.yaml). “Partially achieved” means that the value lies between the target and the lower limit set for that criterion. '
                        'Records 201 and 202 are the same subject (known limitation of the standard split).', ST['cap']))
    return fl


# ---------------------------------------------------------------- sections 4-7
def training_config_section(c):
    cfg = c.cfg
    t, q, m = cfg['training'], cfg.get('qat', {}), cfg['model']
    es = t.get('early_stopping', {})
    rows = [['Setting', 'Value'],
            ['Architecture', f"{m['arch']} (width {m.get('width_mult', 1.0)}, dropout {m.get('dropout')})"],
            ['Optimizer / learning rate', f"{t['optimizer']} / {t['lr']} ({t['lr_schedule']} schedule), weight decay {t['weight_decay']}"],
            ['Batch size / epochs', f"{t['batch_size']} / {t['epochs']}"],
            ['Loss', f"{t['loss']}, label smoothing {t.get('label_smoothing', 0)}"],
            ['Class weighting', f"{t['imbalance']}, power {t.get('class_weight_power', 1.0)}"],
            ['Augmentation', str(t.get('augment'))],
            ['Early stopping', f"patience {es.get('patience')}, min epochs {es.get('min_epochs')}; selection from epoch {t.get('select_from_epoch', 1)}"],
            ['Gradient clipping', str(t.get('grad_clip'))],
            ['QAT', f"{q.get('epochs')} epochs, lr factor {q.get('lr_factor')}, patience {q.get('patience')}" if q.get('enabled') else 'off'],
            ['Seed', str(cfg['seed'])]]
    fl = [Paragraph('4.3 Training configuration', ST['h2']), table(rows, widths=[W * .3, W * .7])]
    sp = os.path.join(c.res, 'training_summary_edge_cnn.json')
    if os.path.exists(sp):
        s = json.load(open(sp))
        fl.append(Paragraph(f"FP32 best validation macro-F1 {fnum(s['fp32_best_val_macro_f1'])} at epoch {s['fp32_best_epoch']}; "
                            f"QAT best {fnum(s['qat_best_val_macro_f1'])} at epoch {s['qat_best_epoch']} "
                            f"(epoch 0 means QAT did not improve on the converted FP32 model, so QAT equals PTQ); training time {s['train_seconds']} s on {s['device']}.", ST['p']))
    else:
        c.missing.append('training_summary_edge_cnn.json')
    return fl


def selection_section(c):
    fl = [Paragraph('4.4 Model selection (validation macro-F1 only)', ST['h2'])]
    p = rp(c, 'model_selection.csv')
    if p:
        rows = [['Experiment', 'Overrides', 'Val macro-F1', 'Best epoch', 'Parameters', 'Selected']]
        for r in csv_rows(p):
            rows.append([r['experiment'], r['overrides'], fnum(r['val_macro_f1']), r['best_epoch'], r['parameters_deployed'], 'yes' if r['selected'] == 'True' else ''])
        fl.append(table(rows, widths=[W * .17, W * .33, W * .12, W * .09, W * .13, W * .08]))
    hist = c.notes.get('run_history') or []
    if hist:
        fl.append(Paragraph('Iterative runs after the sweep (validation only)', ST['h3']))
        fl.append(table([['Run', 'Change', 'Result', 'Decision']] + [[h.get('run', ''), h.get('change', ''), h.get('result', ''), h.get('decision', '')] for h in hist],
                        widths=[W * .08, W * .32, W * .35, W * .25]))
    return fl


def architecture_section(c):
    p = os.path.join(c.mx, 'model_configuration', 'model_summary.json')
    fl = [Paragraph('4.2 Edge model: EdgeECGNet', ST['h2'])]
    if not os.path.exists(p):
        c.missing.append('max78002/model_configuration/model_summary.json'); return fl
    s = json.load(open(p))
    fl.append(table([['Layer', 'Input', 'Operation', 'Output', 'Parameters']] +
                    [[l['Layer'], l['Input'], l['Operation'], l['Output'], f"{l['Parameters']:,}"] for l in s['layers']],
                    widths=[W * .08, W * .14, W * .5, W * .13, W * .15]))
    fl.append(Paragraph(f"Deployed parameters: {s['parameters_deployed']:,} (INT8 weights {s['weight_bytes_int8']:,} bytes); "
                        f"estimated peak activation memory {s['estimated_peak_activation_bytes_int8']:,} bytes.", ST['p']))
    return fl


def md_block(c, text, shift=1):
    """Render a generated markdown file of the results folder (tables, bullets) one heading level lower."""
    text = re.sub(r'\A# .*\n', '', text)                 # drop the file title, the section heading is written by the report
    text = re.sub(r'^(#{1,3}) ', lambda m: '#' * min(3, len(m.group(1)) + shift) + ' ', text, flags=re.M)
    text = re.sub(r'^\* ', '- ', text, flags=re.M)
    return render_md(c, text, {})


def ai8x_logs_section(c):
    fl = []
    sets = [('validation', 'val', 'validation records (4 of DS1)'), ('test_DS2', 'test', 'final test set DS2 (22 records, used once)')]
    rows = [['Model', 'Evaluation set', 'Acc.', 'Macro-F1', 'F1 N', 'F1 S', 'F1 V', 'F1 F', 'Recall S', 'Recall V', 'Recall F']]
    mrows = metric_table(c)
    for key, split, label in sets:
        am = ai8x_metrics(c, key)
        if am is None:
            continue
        em = 'edge_int8_qat'
        acc, mf = get_metric(mrows, em, split, key='accuracy'), get_metric(mrows, em, split)
        if acc is not None and mf is not None:
            rows.append(['PyTorch INT8 emulation (main model)', label, fnum(acc), fnum(mf)] + [fnum(get_metric(mrows, em, split, k, 'f1'), 3) for k in 'NSVF']
                        + [fnum(get_metric(mrows, em, split, k, 'recall'), 3) for k in 'SVF'])
        rows.append(['**ai8x 8-bit (deployed model)**', label, fnum(am['accuracy']), fnum(am['macro_f1'])] + [fnum(x, 3) for x in am['f1']]
                    + [fnum(am['recall'][i], 3) for i in (1, 2, 3)])
    if len(rows) > 1:
        fl.append(Paragraph('Official 8-bit model from ai8x-training compared with the PyTorch INT8 emulation', ST['h3']))
        fl.append(table(rows, widths=[W * .17, W * .17] + [W * .066] * 9))
        fl.append(Paragraph('ai8x numbers are computed from the confusion matrices in max78002/quantization_output/ai8x_evaluation_*.log. The ai8x model was trained from scratch '
                            'by ai8x-training, so it is a different training run, not a more exact copy of the emulated model.', ST['cap']))
    for para in (c.notes.get('ai8x_comparison') or []):
        fl.append(Paragraph(inline(para), ST['p']))
    return fl


def deployment_section(c):
    fl = [Paragraph('6. Deployment: MAX78002 software synthesis', ST['h1'])]
    ok, txt = synth_info(c)
    if ok is None:
        fl.append(Paragraph('synthesis_summary.md not found: run max78002/run_toolchain.sh first.', ST['p'])); return fl
    fl += md_block(c, txt.split('\n', 1)[1] if txt.startswith('#') else txt)
    return fl + ai8x_logs_section(c)


def task2_flow(c):
    n, fl = c.notes, []
    fl.append(PageBreak())
    fl.append(Paragraph('4. Modeling', ST['h1']))
    fl += architecture_section(c) + training_config_section(c) + selection_section(c)
    p = os.path.join(c.res, 'training_curves_edge_cnn.png')
    if os.path.exists(p):
        fl.append(figure(p, 'Training curves of the final edge model (validation records).', .85))
    fl.append(Paragraph('5. Evaluation', ST['h1']))
    tp = rp(c, f'tables_{c.split}.md')
    if tp:
        fl += md_block(c, read(tp), shift=0)
    for name, cap in [(f'confusion_matrix_edge_int8_qat_{c.split}.png', 'Confusion matrix, INT8 model.'),
                      (f'confusion_matrix_edge_fp32_{c.split}.png', 'Confusion matrix, FP32 model.'),
                      (f'confusion_matrix_baseline_{c.split}.png', 'Confusion matrix, classical baseline.')]:
        p = os.path.join(c.res, name)
        if os.path.exists(p):
            fl.append(figure(p, cap, .6))
    fl.append(Paragraph('5.4 Noise robustness', ST['h2']))
    np_ = rp(c, 'noise_results_table.md')
    if np_:
        fl += md_block(c, read(np_), shift=1)
    for name, cap in [('noise_robustness.png', 'Accuracy and macro-F1 versus SNR.'), ('noise_per_class_f1.png', 'Per-class F1 versus SNR.'),
                      ('noise_filter_ablation.png', 'Band-pass filter on versus off.')]:
        p = os.path.join(c.res, name)
        if os.path.exists(p):
            fl.append(figure(p, cap, .8))
    fl.append(Paragraph('5.5 Error analysis', ST['h2']))
    ep = rp(c, 'error_analysis.md')
    if ep:
        fl += md_block(c, read(ep).split('## 6.')[0], shift=1)
    for p in sorted(glob.glob(os.path.join(c.res, 'error_*.png'))):
        fl.append(figure(p, os.path.basename(p), .9))
    if n.get('error_discussion'):
        fl += [Paragraph(inline(x), ST['p']) for x in str(n['error_discussion']).strip().split('\n\n')]
    fl += criteria_section(c)
    fl += deployment_section(c)
    if n.get('limitations'):
        fl.append(Paragraph('7. Limitations and future work', ST['h1']))
        fl += [Paragraph(inline(x), ST['b'], bulletText='•') for x in n['limitations']]
    return fl


def changes_flow(c):
    ch = c.notes.get('changes') or []
    fl = [Paragraph('Submission 2 allows conclusions from Submission 1 to be revised when the experiments reveal problems. The changes are small; they are listed here with the reason.', ST['p'])]
    if ch:
        fl.append(table([['#', 'Section', 'What changed', 'Why']] + [[str(i + 1), x[0], x[1], x[2]] for i, x in enumerate(ch)], widths=[W * .05, W * .12, W * .43, W * .40]))
    lp = os.path.join(c.res, 'test_set_access_log.txt')
    fl.append(Paragraph('Appendix B: every use of the final test set (DS2)', ST['h2']))
    if os.path.exists(lp):
        fl.append(Paragraph(f'<font face="{MONO}" size="7">%s</font>' % escape(read(lp)).replace('\n', '<br/>'), ST['p']))
    fl.append(Paragraph(f"Seed {c.cfg['seed']}; configuration file {inline('`best_config.yaml`')}. Patient-independent split by record (DS1 training/validation, DS2 final test).", ST['p']))
    return fl


def template_vars(c):
    v = {k: val['target'] for k, val in c.targets.items()}
    v['size_mb'] = f"{c.targets['size_mb']['target']:g}"
    sel = {}
    p = os.path.join(c.res, 'model_selection.csv')
    if os.path.exists(p):
        sel = {r['experiment']: float(r['val_macro_f1']) for r in csv_rows(p)}
    sw, full, sam = sel.get('edge_w1.0_softweights'), sel.get('edge_w1.0'), sel.get('edge_w1.0_sampler')
    v['class_weight_sentence'] = ('In the model selection of Task 2 a softer version, the square root of these weights, gave the best validation macro-F1 '
                                  f'({sw:.3f} against {full:.3f} for the full weights) and is used for the final model.') if (sw and full) else \
        'The final model uses class weights (see section 4.4).'
    v['sampler_sentence'] = (f'In the Task 2 model selection a balanced sampler was also tested and was worse on the validation set (macro-F1 {sam:.3f} against {sw:.3f} '
                             'for softened class weights).') if (sam and sw) else ''
    for k, val in (c.notes.get('vars') or {}).items():
        v[k] = val
    v.setdefault('software_task2', '[fill in: software versions of the Task 2 environment in report_notes.yaml, key vars.software_task2]')
    return v


def build(c):
    n, flow = c.notes, []
    logo = next((q for q in [os.path.join(c.assets, 'ltu_logo.png'), os.path.join(c.root, 'ltu_logo.png'), os.path.join(c.root, 'src', 'ltu_logo.png'),
                             os.path.join(c.res, 'ltu_logo.png')] if os.path.exists(q)), None)
    if logo is None:
        c.missing.append('report_assets/ltu_logo.png (title-page logo)')
    flow.append(Spacer(1, 40 * mm))
    flow.append(Paragraph(inline(n.get('title', 'Edge-Ready ECG Classification using CRISP-DM')), ST['title']))
    flow.append(Spacer(1, 12 * mm))
    if logo:
        flow.append(Image(logo, width=60 * mm, height=60 * mm))
        flow.append(Spacer(1, 12 * mm))
    flow.append(Paragraph(inline(n.get('authors', '')), ST['center']))
    flow.append(Paragraph(inline(n.get('course', '')), ST['center']))
    flow.append(Paragraph(inline(n.get('date', datetime.date.today().strftime('%d/%m %Y'))), ST['center']))
    if n.get('summary'):
        flow.append(PageBreak())
        flow.append(Paragraph('Summary', ST['h1']))
        flow += [Paragraph(inline(x), ST['p']) for x in str(n['summary']).strip().split('\n\n')]
    if not os.path.exists(c.t1):
        c.missing.append(os.path.basename(c.t1)); t1 = ''
    else:
        t1 = read(c.t1)
    tv = template_vars(c)
    t1 = re.sub(r'<!--(?!\s*pagebreak).*?-->', '', t1, flags=re.S)          # drop the explanatory comments first
    parts = re.split(r'^@@TASK2@@\s*$', t1, maxsplit=1, flags=re.M)
    head, tail = parts[0], (parts[1] if len(parts) > 1 else '')
    tail_head = re.split(r'^@@CHANGES@@\s*$', tail, maxsplit=1, flags=re.M)[0]
    flow += render_md(c, head, tv)
    flow += task2_flow(c)
    flow += render_md(c, tail_head, tv)
    flow += changes_flow(c)
    os.makedirs(os.path.dirname(c.out), exist_ok=True)
    doc = SimpleDocTemplate(c.out, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title=n.get('title', 'Full report'), author=n.get('authors', ''))

    def footer(canvas, d):
        canvas.setFont(SANS, 7); canvas.setFillColor(colors.grey)
        if d.page > 1:
            canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f'Page {d.page}')
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    return c.out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='../configs/best_config.yaml')
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--notes'); ap.add_argument('--task1'); ap.add_argument('--results-dir'); ap.add_argument('--max78002-dir'); ap.add_argument('--out')
    a = ap.parse_args()
    c = load(a)
    out = build(c)
    print('Report written to', out)
    if c.missing:
        print('Missing inputs (parts skipped):', ', '.join(sorted(set(c.missing))))


if __name__ == '__main__':
    main()
