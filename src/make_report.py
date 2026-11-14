"""Build ONE report (Task 1 + Task 2) as a PDF and as a Word document from the files of the project.

Usage (from src/):
    python export_eda_figures.py                                   # once, after the notebook has been run and saved with outputs
    python make_report.py --config ../configs/best_config.yaml --split test                 # PDF and Word
    python make_report.py --config ../configs/best_config.yaml --split test --format docx   # Word only
Output: ../report/Full_report.pdf and ../report/Full_report.docx
        (needs: pip install reportlab pyyaml matplotlib python-docx)
The Word file is written by docx_writer.py from exactly the same content as the PDF.

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
    c.docx_out = args.docx_out or os.path.splitext(c.out)[0] + '.docx'
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
    fl = [Paragraph('5.6 Success-Criteria Review', ST['h2'])]
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
    fl = [Paragraph('4.3 Training Configuration', ST['h2']), table(rows, widths=[W * .3, W * .7])]
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
    fl = [Paragraph('4.4 Model Selection (validation macro-F1 only)', ST['h2'])]
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


def baseline_section(c):
    """4.1 Baselines: classical models (baseline.py) and the very small CNN, all scored on the VALIDATION records."""
    fl = [Paragraph('4.1 Baseline', ST['h2'])]
    fl.append(Paragraph(inline('Three simple reference models were trained on the 18 training records and scored on the 4 validation records only '
                        '(DS2 was not used to choose any of them). The two classical baselines (`baseline.py`) use the 360 normalised samples '
                        'of a beat as features: **Logistic Regression** (multinomial, `class_weight=balanced`, C = 1.0) and **Random Forest** '
                        '(200 trees, minimum 2 samples per leaf, `class_weight=balanced_subsample`). The better of the two on validation macro-F1 '
                        'is kept as "the baseline" in all later tables. The third reference is a **very small CNN** (`tiny_cnn`, 3 convolution layers) '
                        'trained with exactly the same recipe as the edge model, to show what a minimal deep-learning model achieves.'), ST['p']))
    rows = [['Model', 'Type', 'Parameters', 'Accuracy', 'Macro-F1', 'F1 N', 'F1 S', 'F1 V', 'F1 F', 'Kept as baseline']]
    best = {}
    bp = os.path.join(c.root, 'models', 'baseline_best.json')
    if os.path.exists(bp):
        best = json.load(open(bp))
    else:
        c.missing.append('models/baseline_best.json')
    mrows = metric_table(c)
    bv = os.path.join(c.res, 'baseline_validation.csv')
    if os.path.exists(bv):
        vrows = csv_rows(bv)
        params = {'baseline_logreg': '1,444 (weights + biases)',
                  'baseline_random_forest': next((f"{int(float(r['parameters'])):,} (tree nodes)" for r in mrows
                                                  if r['model'] == 'baseline' and r['split'] == 'val' and r['class'] == 'overall'
                                                  and r.get('parameters')), 'tree nodes')}
        names = {'baseline_logreg': ('Logistic Regression', 'classical'), 'baseline_random_forest': ('Random Forest', 'classical')}
        for key, (name, kind) in names.items():
            g = lambda cls, k: get_metric(vrows, key, 'val', cls, k)
            if g('overall', 'macro_f1') is None:
                continue
            rows.append([name, kind, params[key], fnum(g('overall', 'accuracy')), fnum(g('overall', 'macro_f1'))]
                        + [fnum(g(k, 'f1'), 3) for k in 'NSVF'] + ['**yes**' if best.get('best') == key.replace('baseline_', '') else ''])
    else:
        c.missing.append('baseline_validation.csv')
    if get_metric(mrows, 'tiny_cnn_fp32', 'val') is not None:
        g = lambda cls, k: get_metric(mrows, 'tiny_cnn_fp32', 'val', cls, k)
        tp = next((f"{int(float(r['parameters'])):,}" for r in mrows if r['model'] == 'tiny_cnn_fp32' and r['split'] == 'val'
                   and r['class'] == 'overall' and r.get('parameters')), '')
        rows.append(['Tiny CNN', 'deep learning (FP32)', tp, fnum(g('overall', 'accuracy')), fnum(g('overall', 'macro_f1'))]
                    + [fnum(g(k, 'f1'), 3) for k in 'NSVF'] + ['reference'])
    if len(rows) > 1:
        fl.append(table(rows, widths=[W * .14, W * .11, W * .15, W * .105, W * .105, W * .06, W * .06, W * .06, W * .06, W * .11]))
        fl.append(Paragraph('All values on the validation records 215, 220, 223 and 230 (10,266 beats, only 15 of class F). Results of the kept baseline on the final '
                            'test set DS2 are in sections 5.1 and 5.2.', ST['cap']))
        rf = get_metric(csv_rows(bv), 'baseline_random_forest', 'val', key='accuracy') if os.path.exists(bv) else None
        rfm = get_metric(csv_rows(bv), 'baseline_random_forest', 'val') if os.path.exists(bv) else None
        rfs = get_metric(csv_rows(bv), 'baseline_random_forest', 'val', 'S', 'recall') if os.path.exists(bv) else None
        if rf is not None and rfm is not None and rfs is not None:
            fl.append(Paragraph(f'The Random Forest has the highest accuracy ({rf:.3f}) but a low macro-F1 ({rfm:.3f}): it predicts almost every beat '
                                f'as N, finds only {100 * rfs:.0f} % of the S beats and no F beat. This is why accuracy is never used alone in this '
                                'report. The baselines set the bar that the edge model must beat on macro-F1 and on the minority classes '
                                '(success criterion 1.4), and the Random Forest could not run on the MAX78002 CNN accelerator in any case.', ST['p']))
    for para in (c.notes.get('baseline_discussion') or []):
        fl.append(Paragraph(inline(para), ST['p']))
    return fl


def _agreement(c, a, b, split):
    """Share of beats that two models give the same class (from results/preds/<model>_<split>.npy), or None."""
    pa, pb = (os.path.join(c.res, 'preds', f'{m}_{split}.npy') for m in (a, b))
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return None
    import numpy as np
    x, y = np.load(pa), np.load(pb)
    return float((x == y).mean()) if x.shape == y.shape else None


def quantization_section(c):
    """4.5 Quantization: why QAT, how it was done (PyTorch emulation and official ai8x flow), and its effect (FP32 vs INT8)."""
    q, t = c.cfg.get('qat', {}), c.cfg['training']
    fl = [Paragraph('4.5 Quantization', ST['h2'])]
    fl.append(Paragraph('Why QAT is needed', ST['h3']))
    fl.append(Paragraph(inline(
        'The MAX78002 CNN accelerator does not compute in floating point. It stores every weight and bias as a signed 8-bit number '
        'with one power-of-two scale per layer, it has no BatchNorm layer, and it rounds and clips the output of every layer to 8 bit '
        '(range [-1, 127/128] in steps of 1/128). A model trained in FP32 never saw these rounding and clipping errors, and in a network '
        'of six convolution layers they add up. **Quantization-aware training (QAT)** puts the same rounding into the forward pass '
        'during training, so the weights can adapt to it. It is compared with plain **post-training quantization (PTQ)**, which only '
        'rounds the finished FP32 model.'), ST['p']))
    fl.append(Paragraph('How it was performed', ST['h3']))
    steps = [
        'Start from the best FP32 weights (section 4.3).',
        'Fold BatchNorm into the convolution weights and biases and freeze its statistics, because the chip has no BatchNorm layer (as in ai8x).',
        'Weights and biases: rounded to 8 bit with a power-of-two scale per layer.',
        'Input and every activation: rounded to 8 bit with step 1/128 and clipped to [-1, 127/128]; the last linear layer keeps a wide 32-bit output.',
        'The rounding is non-differentiable, so gradients pass through it unchanged (**straight-through estimator**).',
        f"Fine-tuning: up to {q.get('epochs', '?')} epochs with learning rate {t.get('lr')} × {q.get('lr_factor', '?')} = "
        f"{t.get('lr', 0) * q.get('lr_factor', 0):g}, early stopping with patience {q.get('patience', '?')}.",
        'The best QAT epoch is chosen on the **validation** macro-F1 measured with the 8-bit emulation switched on; '
        'epoch 0 is the quantized FP32 model before any QAT update (= PTQ).',
    ]
    fl += [Paragraph(inline(x), ST['b'], bulletText='•') for x in steps]
    pol = os.path.join(c.mx, 'model_configuration', 'policies', 'qat_policy_ecg.yaml')
    sh = os.path.join(c.mx, 'model_configuration', 'scripts', 'train_ecg.sh')
    if os.path.exists(pol):
        start = (yaml.safe_load(read(pol)) or {}).get('start_epoch')
        ep = re.search(r'--epochs\s+(\d+)', read(sh)) if os.path.exists(sh) else None
        fl.append(Paragraph(inline(
            'The PyTorch version above is an **emulation** and is not bit-exact with the chip. The model that is synthesized for the MAX78002 '
            f"(section 6) is trained separately with the official **ai8x-training** QAT: FP32 until epoch {start}, then 8-bit QAT"
            + (f" up to epoch {ep.group(1)}" if ep else '') + ' (`qat_policy_ecg.yaml`), followed by `quantize.py` and the 8-bit evaluation.'), ST['p']))
    fl.append(Paragraph('Did quantization change the performance?', ST['h3']))
    mrows = metric_table(c)
    rows = [['Evaluation set', 'Model', 'Accuracy', 'Macro-F1', 'Change vs FP32', 'Same class as FP32']]
    names = [('edge_fp32', 'FP32'), ('edge_int8_ptq', 'INT8 PTQ (emulated)'), ('edge_int8_qat', 'INT8 QAT (emulated)')]
    splits = ['val'] + ([c.split] if c.split != 'val' else [])
    labels = {'val': 'validation (DS1)', 'test': 'test (DS2)'}
    for sp in splits:
        f32 = get_metric(mrows, 'edge_fp32', sp)
        for key, name in names:
            mf, acc = get_metric(mrows, key, sp), get_metric(mrows, key, sp, key='accuracy')
            if mf is None:
                continue
            ag = _agreement(c, 'edge_fp32', key, sp) if key != 'edge_fp32' else None
            rows.append([labels[sp], name, fnum(acc), fnum(mf),
                         '–' if key == 'edge_fp32' or f32 is None else f'{mf - f32:+.4f}',
                         '–' if key == 'edge_fp32' else (f'{100 * ag:.1f} %' if ag is not None else 'n/a')])
        am = ai8x_metrics(c, 'validation' if sp == 'val' else 'test_DS2')
        if am:
            rows.append([labels[sp], 'INT8 ai8x (official, separate training run)', fnum(am['accuracy']), fnum(am['macro_f1']),
                         f"{am['macro_f1'] - f32:+.4f} *" if f32 is not None else '–', 'n/a'])
    if len(rows) > 1:
        fl.append(table(rows, widths=[W * .17, W * .3, W * .11, W * .11, W * .14, W * .17]))
        fl.append(Paragraph('Same model architecture and parameter count (25,268) in every row; INT8 needs one byte per weight. '
                            '"Same class as FP32" is the share of beats that get the same prediction as the FP32 model. '
                            '* The ai8x model was trained from scratch in a separate run, so its difference to FP32 is not caused by quantization alone.', ST['cap']))
    sp_ = os.path.join(c.res, 'training_summary_edge_cnn.json')
    s = json.load(open(sp_)) if os.path.exists(sp_) else {}
    f32v, ptqv, qatv = (get_metric(mrows, k, c.split) for k in ('edge_fp32', 'edge_int8_ptq', 'edge_int8_qat'))
    txt = []
    if f32v is not None and qatv is not None:
        txt.append(f"On the {labels.get(c.split, c.split)} set, 8-bit quantization changes macro-F1 by {qatv - f32v:+.4f} "
                   f"(FP32 {f32v:.4f}, INT8 {qatv:.4f})" + (
                       f", which is within the target of at most {tgt} from section 1.4" if (tgt := c.targets.get('int8_drop', {}).get('target')) is not None
                       and f32v - qatv <= tgt else (f", which is more than the target of at most {tgt} from section 1.4" if tgt is not None else '')))
    if s.get('qat_best_epoch') == 0:
        agq = _agreement(c, 'edge_int8_ptq', 'edge_int8_qat', 'val')
        txt.append('The best QAT epoch was **epoch 0**: no QAT update improved the validation macro-F1 of the quantized starting model. '
                   'The saved QAT model is therefore the PTQ model' + (f' (identical predictions on {100 * agq:.0f} % of the validation beats)' if agq is not None else '')
                   + ', and in this run QAT gave no measurable gain over plain post-training quantization. QAT is still kept in the pipeline because it '
                   'is required for the official ai8x flow and protects against larger losses if the network or the data change.')
    elif s.get('qat_best_epoch') and ptqv is not None and qatv is not None:
        txt.append(f"QAT selected epoch {s['qat_best_epoch']} and changed macro-F1 by {qatv - ptqv:+.4f} compared with PTQ.")
    if txt:
        fl.append(Paragraph(inline('. '.join(x.rstrip('.') for x in txt) + '.'), ST['p']))
    for para in (c.notes.get('quantization_discussion') or []):
        fl.append(Paragraph(inline(para), ST['p']))
    fl.append(Paragraph(inline('The per-class FP32-versus-INT8 comparison on the final test set is in section 5.3.'), ST['p']))
    return fl


def architecture_section(c):
    p = os.path.join(c.mx, 'model_configuration', 'model_summary.json')
    fl = [Paragraph('4.2 Edge Model: EdgeECGNet', ST['h2'])]
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


# ---------------------------------------------------------------- explanatory text (written from the numbers)
NAMES = {'baseline': 'classical baseline', 'tiny_cnn_fp32': 'Tiny CNN (FP32)', 'edge_fp32': 'EdgeECGNet FP32',
         'edge_int8_ptq': 'EdgeECGNet INT8 PTQ', 'edge_int8_qat': 'EdgeECGNet INT8 QAT', 'edge_int8_ai8x': 'EdgeECGNet INT8 ai8x'}
SPLIT_NAME = {'val': 'validation records (DS1)', 'test': 'final test set DS2'}
CLS = ['N', 'S', 'V', 'F']


def synth_parts(txt):
    """synthesis_summary.md (collect_evidence.py) -> {heading: text}."""
    parts = {}
    for chunk in re.split(r'(?m)^## ', txt)[1:]:
        head, _, body = chunk.partition('\n')
        parts[head.strip()] = body
    return parts


def deployment_section(c):
    fl = [Paragraph('6. Deployment: MAX78002 software synthesis', ST['h1'])]
    ok, txt = synth_info(c)
    parts = synth_parts(txt) if txt else {}
    get = lambda key: next((v for k, v in parts.items() if k.lower().startswith(key.lower())), None)
    ms_p = os.path.join(c.mx, 'model_configuration', 'model_summary.json')
    ms = json.load(open(ms_p)) if os.path.exists(ms_p) else {}

    # 6.1
    fl.append(Paragraph('6.1 MAX78002 Compatibility', ST['h2']))
    fl.append(P('The architecture would in theory run on the MAX78002 CNN accelerator because the model uses INT8 weights, has '
                'hardware-supported layers, and is estimated to fit into the memory and data requirements. The layer types are 1D '
                'convolution (kernel 5, 3 and 1), max pooling before the convolution, ReLU, BatchNorm folded into the weights, and one '
                'linear output layer (section 4.2).'))
    for key, title in (('Model', 'Model'), ('Resources reported by ai8xize', 'Resources reported by ai8xize')):
        body = get(key)
        if body:
            fl.append(H(title))
            fl += md_block(c, body, shift=0)
    if ok is None:
        fl.append(P('synthesis_summary.md not found: run max78002/run_toolchain.sh first.'))

    # 6.2
    fl.append(Paragraph('6.2 MAX78002 Toolchain', ST['h2']))
    fl.append(P('The Analog Devices flow (`max78002/run_toolchain.sh`): `export_ai8x.py` writes the ai8x model, dataset and network files '
                '-> **ai8x-training** trains the network with quantization-aware training -> `quantize.py` turns the checkpoint into 8 bit '
                '-> the 8-bit model is evaluated -> **ai8xize.py** (ai8x-synthesis) generates the C project for the MAX78002 -> '
                '`collect_evidence.py` writes the summary used here.'))
    for key, title in (('Generated C project', 'Generated C project'), ('Quantization', 'Quantization (8-bit) tensors from quantize.py')):
        body = get(key)
        if body:
            fl.append(H(title))
            fl += md_block(c, body, shift=0)
    fl += ai8x_logs_section(c)

    # 6.3
    fl.append(Paragraph('6.3 Deployment Evidence', ST['h2']))
    files = re.findall(r'^\|\s*(\S+\.[ch])\s*\|\s*(yes|NO)\s*\|', txt or '', flags=re.M)
    wm = re.search(r'Weight memory:\s*([\d,]+) bytes of ([\d,]+) bytes \(([\d.]+)%\)', txt or '')
    quant_ok = bool(re.search(r'^\|\s*\S+\.op\.weight', txt or '', flags=re.M)) or bool(glob.glob(os.path.join(c.mx, 'quantization_output', 'ai8x_evaluation_*.log')))
    pf = lambda b: 'PASS' if b else ('not run' if ok is None else 'FAIL')
    act = ms.get('estimated_peak_activation_bytes_int8')
    rows = [['Requirement', 'Result', 'Evidence'],
            ['Model accepted by MAX78002 toolchain', pf(ok), 'ai8xize.py synthesis log'],
            ['Quantization completed', pf(quant_ok), 'quantize.py: every tensor stored with 8 bits'],
            ['Synthesis completed', pf(ok), 'synthesis_summary.md'],
            ['C/device files generated', pf(bool(files) and all(v == 'yes' for _, v in files)), ', '.join(f for f, _ in files) or '-'],
            ['Weight-memory requirement', pf(bool(wm)), f"{wm.group(1)} of {wm.group(2)} bytes ({wm.group(3)} %)" if wm else '-'],
            ['Data-memory requirement', pf(act is not None), f"estimated peak activations {act:,} bytes (8 bit)" if act else '-']]
    fl.append(table(rows, widths=[W * .36, W * .12, W * .52]))

    # 6.4
    fl.append(Paragraph('6.4 Hardware Limitations', ST['h2']))
    fl.append(P('No MAX78002 board is used in this project, so real latency, energy and RAM use are not measured. The operation count '
                'above is a simulation-level quantity only. This means that while software verification is possible, verification in '
                'terms of hardware in this scenario is not.'))
    fl.append(table([['Software verified (this project)', 'Not hardware verified'],
                     ['Layers and memory fit the MAX78002 rules (model.py check and ai8xize)', 'Inference latency and real-time performance'],
                     ['8-bit quantization effect (PyTorch emulation and ai8x evaluation)', 'Power and energy per inference'],
                     ['C code generated by ai8xize', 'Real accuracy on the chip'],
                     ['Operation count per inference (simulation)', 'Real RAM use and the input pipeline (ADC, R-peak detection, filtering)']],
                    widths=[W * .5, W * .5]))

    # 6.5
    fl.append(Paragraph('6.5 Proposed Hardware Validation', ST['h2']))
    fl.append(P('If a MAX78002 kit became available, we could test the model in several ways. For loading the generated model, the '
                'generated C files could be compiled, and the resulting firmware would in turn be put onto the MAX78002 board. Moreover, to '
                'supply identical inputs for the hardware and Python environment, the input file generated by the toolchain could be fed to '
                'the accelerator. A match of the outputs would confirm that the hardware works correctly. If the output is wrong, the '
                'debugging tools that come with the MAX78002 board would prove useful, for example to compare intermediate layer outputs. '
                'Additionally, with an actual, physical MAX78002 board, the experiment could be conducted with hardware validity and greater '
                'thoroughness by also measuring energy consumption and inference time. The energy consumption could be measured by isolating '
                'the accelerator and measuring the power draw, while the inference time could be measured as the execution time of the CNN '
                'on hardware level.'))
    return fl


def discussion_values(c):
    m, sp = metric_table(c), c.split
    v = {k: get_metric(m, 'edge_int8_qat', sp, k, 'recall') for k in CLS}
    v.update(tiny=get_metric(m, 'tiny_cnn_fp32', sp), main=get_metric(m, 'edge_int8_qat', sp), fp32=get_metric(m, 'edge_fp32', sp))
    am = ai8x_metrics(c, 'test_DS2' if sp == 'test' else 'validation')
    v['ai8x'] = am['macro_f1'] if am else None
    v['nclean'], v['n0'] = noise_value(c, 'edge_int8_qat', 'clean'), noise_value(c, 'edge_int8_qat', 0)
    ms_p = os.path.join(c.mx, 'model_configuration', 'model_summary.json')
    ms = json.load(open(ms_p)) if os.path.exists(ms_p) else {}
    v['params'] = ms.get('parameters_deployed', 25268)
    v['tiny_params'] = next((int(float(r['parameters'])) for r in m if r['model'] == 'tiny_cnn_fp32' and r['class'] == 'overall'
                             and r.get('parameters')), 2900)
    v['records'], v['share'] = None, None
    ep = os.path.join(c.res, 'error_analysis.md')
    if os.path.exists(ep):
        t = read(ep)
        mo = re.search(r'The 3 records with most errors hold (\d+)%', t)
        v['share'] = mo.group(1) if mo else None
        sec = t.split('## 3.')[1] if '## 3.' in t else ''
        recs = re.findall(r'^\|\s*(\d{3})\s*\|', sec, flags=re.M)[:3]
        v['records'] = recs or None
    return v


def final_discussion(c):
    v = discussion_values(c)
    f = lambda x, d=4: '-' if x is None else f'{x:.{d}f}'
    tq, tn = c.targets.get('int8_drop', {}).get('target'), c.targets.get('noise_drop_0db', {}).get('target')
    on_set = 'DS2' if c.split == 'test' else 'the validation records'
    fl = [Paragraph('7. Final Discussion', ST['h1'])]
    qa = []
    qa.append(('What did we learn from the data?', [
        'One of the main things we learned from the data is how challenging it is to classify ECG heartbeats when the classes are highly '
        'imbalanced. Normal heartbeats (N) make up the majority of the dataset, while classes S and F are much more underrepresented. This '
        'can make it difficult for the model to learn meaningful patterns for the less common classes. We also learned that the different '
        'heartbeat classes can have very similar signal patterns, which means that waveform information alone is sometimes not enough to '
        'distinguish between them. In particular, the results suggest that timing information, such as RR intervals, is important when '
        'identifying supraventricular ectopic (S) and fusion (F) beats.']))
    rec_txt = (f" Furthermore, three records ({', '.join(v['records'][:-1])} and {v['records'][-1]}) accounted for {v['share']} % of all "
               "classification errors, showing that performance was also affected by differences between individual ECG recordings."
               if v['records'] and v['share'] else '')
    qa.append(('What was the largest source of error?', [
        f"The largest source of error was the model's difficulty identifying minority classes, particularly S and F. The main model "
        f"achieved a recall of {f(v['V'])} for class V, but only {f(v['S'])} for class S and {f(v['F'])} for class F on {on_set}. This means "
        "that the model was much better at recognizing ventricular ectopic beats than the other abnormal heartbeat types. A big reason is "
        "that the model only receives a one-second heartbeat segment, which does not provide sufficient timing information to reliably "
        "distinguish certain heartbeat types. Class imbalance also contributed to this problem, as the model had much fewer examples of S "
        "and F during training." + rec_txt]))
    p3 = [('The model with the highest accuracy was not necessarily the best model for the Edge-AI application. Accuracy alone does not '
           'provide a complete picture of performance, especially when working with an imbalanced dataset. A model can achieve high '
           'accuracy by correctly classifying the majority class while performing poorly on less common but important classes.')]
    p3.append(f"Our smaller CNN model, with roughly {v['tiny_params']:,} parameters, achieved a macro-F1 score of {f(v['tiny'])}, compared "
              f"to {f(v['main'])} for the main EdgeECGNet INT8 model."
              + (f" The separately trained ai8x model, which was successfully synthesized for the MAX78002, achieved a macro-F1 score of "
                 f"{f(v['ai8x'])} on {on_set}." if v['ai8x'] is not None else '')
              + f" This shows that a larger network does not automatically result in better classification performance. However, EdgeECGNet "
              f"was still a suitable Edge-AI candidate because it contained only {v['params']:,} parameters, required around "
              f"{v['params'] / 1000:.0f} KB for INT8 weights and successfully passed the MAX78002 synthesis process. Therefore, the best "
              "choice depends not only on predictive performance but also on hardware compatibility and deployment requirements.")
    qa.append(('Was the highest-accuracy model also the best Edge-AI model?', p3))
    if v['fp32'] is not None and v['main'] is not None:
        d = v['fp32'] - v['main']
        qa.append(('How much performance was lost after quantization?', [
            f"The performance loss after quantization was relatively small. The FP32 model achieved a macro-F1 score of {f(v['fp32'])}, "
            f"while the INT8 model achieved {f(v['main'])}. This represents a decrease of {f(d)}"
            + (f", which is within our predefined target of a maximum decrease of {tq}" if tq is not None and d <= tq else
               (f", which is more than our predefined target of a maximum decrease of {tq}" if tq is not None else '')) + '. '
            "This suggests that converting the model from 32-bit floating-point calculations to 8-bit integer calculations did not "
            "significantly affect its overall classification performance.",
            'This is an important result for the project because quantization allows the model to use considerably less memory and makes '
            'it more suitable for deployment on resource-constrained Edge-AI hardware such as the MAX78002.',
            'It is important to distinguish between the PyTorch INT8 emulation and the separately trained ai8x 8-bit model. The '
            'quantization comparison above refers to the main model, while the ai8x model represents the model that was actually '
            'synthesized for the MAX78002.']))
    if v['nclean'] is not None and v['n0'] is not None:
        d = v['nclean'] - v['n0']
        qa.append(('How sensitive was the system to ECG noise?', [
            f"The system demonstrated relatively good robustness to ECG noise, with performance decreasing gradually as the noise level "
            f"increased. The main INT8 model achieved a macro-F1 score of around {f(v['nclean'])} under clean conditions and {f(v['n0'])} at "
            f"0 dB signal-to-noise ratio. This represents a decrease of {f(d)}"
            + (f", which is within our predefined maximum acceptable decrease of {tn}" if tn is not None and d <= tn else
               (f", which is more than our predefined maximum acceptable decrease of {tn}" if tn is not None else '')) + '.']))
    qa.append(('What compromises were necessary because of hardware constraints?', [
        'The main compromise was balancing model complexity and classification performance. Since the MAX78002 has limited memory and '
        'computational resources, the network had to remain relatively small and use operations supported by the hardware. This '
        'influenced the architecture and limited the possibility of using more complex models.',
        'Another important compromise was converting the model into INT8. Although quantization caused a small reduction in macro-F1, '
        'it significantly reduced the memory requirements and made deployment more realistic.',
        'We also used individual one-second heartbeat segments as input, which simplified the processing pipeline and kept the model '
        'fairly lightweight. However, this meant that important timing information, particularly RR intervals, was not included. This '
        'limitation negatively affected the classification of S and F beats.',
        'Overall, the project demonstrates that it is possible to create a compact neural network that is compatible with the MAX78002 '
        'toolchain, but achieving high classification performance under strict hardware constraints remains challenging.']))
    qa.append(('What would we change with more time?', [
        'With more time, we would primarily focus on improving the classification of minority classes, like S and F. One possible '
        'improvement would be to include RR intervals and other timing-related features as additional inputs to the model. This could '
        'provide information that is missing from individual one-second heartbeat segments and potentially improve the distinction '
        'between different arrhythmia types.',
        'We would also look at alternative approaches to handling class imbalance, such as different class-weighting strategies, '
        'oversampling and more targeted data augmentation. Additionally, we would experiment with different model architectures and '
        'training configurations to investigate whether better performance could be achieved without significantly increasing the model size.',
        'Another important improvement would be to train and evaluate multiple models using different random seeds. Since the current '
        'INT8 models are based on individual training runs, additional experiments would help determine whether the observed performance '
        'differences are consistent or caused by random variation.',
        'Finally, we would like to test the model on an actual MAX78002 development board. This would allow us to measure real inference '
        'latency, energy consumption and memory usage rather than relying only on software-level synthesis results. Such testing would '
        'provide a more complete understanding of whether the model is suitable for real-world Edge-AI ECG monitoring.']))
    for q, paras in qa:
        fl.append(H(q))
        fl += [P(x) for x in paras]
    return fl + own_text(c, '7')


LIMITS_BEFORE = [
    'Because the model is trained on a small set of patients, the performance of the model heavily depends on which subjects end up '
    'in the test set.',
    'As previously mentioned, there is a severe class imbalance. The N class makes up 89.5 % of the data, while the F class makes up '
    'less than 1 %. Even with the class weights applied during training, the performance for minority classes on the final test set is '
    'less than ideal.',
    'The amplitude varies heavily between records, and three records accounted for the majority of all errors. This shows that the '
    'variability between patients impacts the accuracy of the model.',
    'There is no claim in this report of clinical validity, and there was no prospective clinical testing.',
    'The MIT-BIH recordings are historical (recorded 1975-1979 with Holter recorders); today\'s devices and electrodes differ.',
]
LIMITS_AFTER = [
    'The environments that a mobile or smartwatch device would be in are much noisier than clinical ECG recordings (domain shift to '
    'wearable ECG). The injection of noise and baseline wander into the training data may arguably not be enough to simulate these '
    'real-world conditions.',
]


def limitations_section(c):
    items = LIMITS_BEFORE + list(c.notes.get('limitations') or []) + LIMITS_AFTER
    return [Paragraph('8. Limitations', ST['h1'])] + [B(x) for x in items] + own_text(c, '8')

def P(text):
    return Paragraph(inline(text), ST['p'])


def B(text):
    return Paragraph(inline(text), ST['b'], bulletText='•')


def H(text):
    return Paragraph(inline(text), ST['h3'])


def own_text(c, key):
    """Extra paragraphs written by hand in report_notes.yaml under section_text: {'5.2': [...]} (added after the automatic text)."""
    v = (c.notes.get('section_text') or {}).get(key)
    if not v:
        return []
    return [P(x) for x in ([v] if isinstance(v, str) else v)]


def read_confusion(c, model, split):
    p = os.path.join(c.res, f'confusion_matrix_{model}_{split}.csv')
    if not os.path.exists(p):
        return None
    return [[int(float(r[f'pred_{k}'])) for k in CLS] for r in csv_rows(p)]


def train_counts(c):
    p = os.path.join(c.root, 'data', 'processed', 'dataset_statistics.csv')
    if not os.path.exists(p):
        return None
    r = next((r for r in csv_rows(p) if r.get('Split', '').lower().startswith('train')), None)
    return {k: int(r[k]) for k in CLS} if r else None


def text_42(c):
    p = os.path.join(c.mx, 'model_configuration', 'model_summary.json')
    if not os.path.exists(p):
        return own_text(c, '4.2')
    s = json.load(open(p))
    wb = s['weight_bytes_int8']
    return [P(f"Because the model only has {s['parameters_deployed']:,} parameters, it takes up only a small amount of storage on the device. "
              f"At the same time, the peak activation memory is estimated at a low {s['estimated_peak_activation_bytes_int8']:,} bytes, "
              "which is fitting for an Edge-AI device."),
            H('Why this model fits an Edge-AI chip'),
            B('**Only supported layers.** It uses only layers that the MAX78002 can run: 1D convolution, max pooling, ReLU and one linear layer. '
              'BatchNorm is used during training and is then merged into the convolution weights.'),
            B('**Same layout as the chip.** Pooling comes before each convolution, exactly like the ai8x "fused" layers. So the same model '
              'description is used for training and for the chip (section 6).'),
            B('**Sees the whole beat.** The first layer keeps all 360 samples to see the sharp QRS peak. Each later layer halves the length, '
              'so the deep layers see the whole beat with little computation.'),
            B(f"**Very small.** {wb / 1024:.1f} KB of 8-bit weights is only {100 * wb / (2 * 1024 * 1024):.1f} % of the 2 MB limit, "
              "and the input size is fixed (1 x 360).")
            ] + own_text(c, '4.2')

def text_43(c):
    t = c.cfg['training']
    return [P(f"The table shows the settings of the final model (`best_config.yaml`). Three choices matter most: (1) rare classes get a "
              f"higher weight in the loss (class weights from the training records, softened with power {t.get('class_weight_power', 1.0)}); "
              f"(2) the best epoch is picked by **validation macro-F1**, not by accuracy, because almost all beats are N; "
              f"(3) augmentation is used on training beats only.")] + own_text(c, '4.3')

def text_44(c):
    p = os.path.join(c.res, 'model_selection.csv')
    if not os.path.exists(p):
        return own_text(c, '4.4')
    rows = csv_rows(p)
    sw = os.path.join(c.root, 'configs', 'sweep.yaml')
    margin = (yaml.safe_load(read(sw)) or {}).get('tie_margin', 0.005) if os.path.exists(sw) else 0.005
    f = {r['experiment']: float(r['val_macro_f1']) for r in rows}
    best = max(f.values())
    sel = next((r for r in rows if r['selected'] == 'True'), None)
    near = [k for k, v in f.items() if v >= best - margin and (not sel or k != sel['experiment'])]
    out = [B(f"**Models tried:** {len(rows)} short training runs: the Tiny CNN, and EdgeECGNet with half, normal and 1.5x the number of channels."),
           B('**Settings varied:** the learning rate (0.001 or 0.003) and the way rare classes are handled (class weights, softer class '
             'weights, or a balanced sampler).'),
           B(f"**How the winner was chosen:** highest **validation macro-F1**. If two runs are closer than {margin}, the smaller model wins. "
             "The test set DS2 was not used.")]
    if sel:
        out.append(B(f"**Result:** **{sel['experiment']}** was selected ({float(sel['val_macro_f1']):.4f}, {int(float(sel['parameters_deployed'])):,} "
                     f"parameters)" + (f". {', '.join(near)} scored almost the same ({', '.join(f'{f[k]:.4f}' for k in near)})" if near else '')
                     + f". The bigger model (1.5x, {f.get('edge_w1.5', float('nan')):.4f}) was not better and is twice as large; the smaller "
                     "models scored lower. So the normal width gives the best balance between score and size. The final settings were "
                     "then fixed in the extra runs shown below."))
        out.append(P('Note: the validation set has only 15 F beats, so small differences between runs can be chance.'))
    return out + own_text(c, '4.4')

def text_51(c):
    m, sp = metric_table(c), c.split
    keys = [k for k in NAMES if get_metric(m, k, sp) is not None]
    if not keys:
        return own_text(c, '5.1')
    sup = [get_metric(m, 'edge_fp32' if 'edge_fp32' in keys else keys[0], sp, k, 'support') or 0 for k in CLS]
    n = int(sum(sup))
    share = sup[0] / n if n else 0
    best = max(keys, key=lambda k: get_metric(m, k, sp))
    main = 'edge_int8_qat' if 'edge_int8_qat' in keys else keys[-1]
    out = [P(f"These results are on the {SPLIT_NAME.get(sp, sp)}" + (f" ({n:,} beats" + (" from 22 people the models never saw; it was used only once)" if sp == 'test' else ')') if n else '') + '.')]
    if n:
        out.append(B(f"**Why not only accuracy?** {100 * share:.0f} % of the beats are N. A model that always says \"N\" would get "
                     f"{share:.2f} accuracy but find no abnormal beat at all (macro-F1 only {2 * share / (1 + share) / 4:.2f}). "
                     "Macro-F1 gives each of the four classes the same weight, so it is our main score."))
    out.append(B(f"**Best score:** {NAMES[best]}, macro-F1 {get_metric(m, best, sp):.4f}."))
    out.append(B(f"**Our 8-bit chip model** ({NAMES[main]}): macro-F1 {get_metric(m, main, sp):.4f}, accuracy {get_metric(m, main, sp, key='accuracy'):.4f}."))
    if 'tiny_cnn_fp32' in keys and 'edge_fp32' in keys and get_metric(m, 'tiny_cnn_fp32', sp) > get_metric(m, 'edge_fp32', sp):
        out.append(B(f"**Honest note:** the Tiny CNN ({get_metric(m, 'tiny_cnn_fp32', sp):.4f}) beats EdgeECGNet ({get_metric(m, 'edge_fp32', sp):.4f}) "
                     "here, although EdgeECGNet was better on the validation records. With only four validation records, the choice of model was uncertain."))
    return out + own_text(c, '5.1')

def text_52(c):
    m, sp = metric_table(c), c.split
    main = next((k for k in ('edge_int8_qat', 'edge_fp32') if get_metric(m, k, sp) is not None), None)
    if not main:
        return own_text(c, '5.2')
    g = lambda k, key: get_metric(m, main, sp, k, key)
    cm = read_confusion(c, main, sp)
    tc = train_counts(c)
    out = [H('Which classes are difficult, and why?'),
           P(f"Answer for our chip model ({NAMES[main]}). Recall = how many beats of a class are found. Precision = how many of the "
             f"beats the model calls that class are really that class.")]
    pct = lambda row, col: 100 * cm[row][col] / max(sum(cm[row]), 1)
    out.append(B(f"**N (normal) - easy.** F1 {g('N', 'f1'):.2f}, recall {g('N', 'recall'):.2f}. About 90 % of all beats are N, so the model has "
                 "seen very many examples."))
    if cm:
        out.append(B(f"**V (ventricular) - medium.** F1 {g('V', 'f1'):.2f}. The model finds {100 * g('V', 'recall'):.0f} % of the V beats, "
                     f"because V beats look clearly different (wide, strange QRS). But it also calls {cm[0][2]:,} normal beats \"V\", so "
                     f"precision is only {g('V', 'precision'):.2f}. Many of these are bundle-branch-block beats (L, R): they are labelled N "
                     "but also have a wide QRS, so they look like V (section 5.5)."))
        lab = {k: v for k, v in zip(sorted(['S', 'F'], key=lambda k: -g(k, 'f1')), ['hard', 'hardest'])}
        out.append(B(f"**S (supraventricular) - {lab['S']}.** F1 {g('S', 'f1'):.2f}. The model finds only {100 * g('S', 'recall'):.0f} % of the S beats; "
                     f"{pct(1, 0):.0f} % are called N and {pct(1, 2):.0f} % V. The reason: an S beat has almost the same shape as a normal beat. "
                     "What makes it S is that it comes **too early**. Our input is one 1-second beat without the time to the previous beat "
                     "(RR interval), so the model cannot see this."))
        out.append(B(f"**F (fusion) - {lab['F']}.** F1 {g('F', 'f1'):.2f}, recall {g('F', 'recall'):.2f}. An F beat is a mix of a normal and a "
                     f"ventricular beat, so it looks half N, half V ({pct(3, 0):.0f} % are called N and {pct(3, 2):.0f} % V). It is also very "
                     "rare" + (f": only {tc['F']} F beats in the training data" if tc else '') + '.'))
    out.append(P('**In short:** classes S and F are especially difficult because of the class imbalance and the variability between '
                 'patients - not to mention that S and F beats are distinguished mainly by timing (RR interval), which a one-second '
                 'window does not give us.'))
    return out + own_text(c, '5.2')


def quant_compare(c):
    """5.3: FP32 vs INT8 per class (table + figure) and the discussion."""
    m, sp = metric_table(c), c.split
    keys = [k for k in ('edge_fp32', 'edge_int8_ptq', 'edge_int8_qat') if get_metric(m, k, sp) is not None]
    if len(keys) < 2 or 'edge_fp32' not in keys:
        return own_text(c, '5.3')
    q = 'edge_int8_qat' if 'edge_int8_qat' in keys else keys[-1]
    f32, i8 = get_metric(m, 'edge_fp32', sp), get_metric(m, q, sp)
    ag = _agreement(c, 'edge_fp32', q, sp)
    out = [P(f"**FP32 macro-F1:** {f32:.4f}   **INT8 macro-F1:** {i8:.4f}   **Difference:** {i8 - f32:+.4f}")]
    rows = [['Class', 'Recall FP32', 'Recall INT8', 'Precision FP32', 'Precision INT8', 'F1 FP32', 'F1 INT8', 'F1 change']]
    for k in CLS:
        g = lambda mod, key: get_metric(m, mod, sp, k, key)
        rows.append([k, fnum(g('edge_fp32', 'recall'), 3), fnum(g(q, 'recall'), 3), fnum(g('edge_fp32', 'precision'), 3),
                     fnum(g(q, 'precision'), 3), fnum(g('edge_fp32', 'f1'), 3), fnum(g(q, 'f1'), 3), f"{g(q, 'f1') - g('edge_fp32', 'f1'):+.3f}"])
    rows.append(['**Macro-F1**', '', '', '', '', f'{f32:.4f}', f'{i8:.4f}', f'{i8 - f32:+.4f}'])
    out.append(table(rows, widths=[W * .16] + [W * .12] * 7))
    out.append(Paragraph(f"FP32 versus 8-bit ({NAMES[q]}, PyTorch emulation) on the {SPLIT_NAME.get(sp, sp)}, per class. Same 25,268 parameters; "
                         "INT8 stores one byte per weight." + (f" {100 * ag:.1f} % of the beats get the same class from both models." if ag is not None else ''), ST['cap']))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        series = [(NAMES[k], [get_metric(m, k, sp, x, 'f1') for x in CLS] + [get_metric(m, k, sp)]) for k in keys]
        am = ai8x_metrics(c, 'test_DS2' if sp == 'test' else 'validation')
        if am:
            series.append(('INT8 ai8x (separate run)', list(am['f1']) + [am['macro_f1']]))
        fig, ax = plt.subplots(figsize=(8, 3.4), dpi=200)
        w = 0.8 / len(series)
        cols = ['#1f4e79', '#5b8cc0', '#c0392b', '#e67e22']
        for i, (name, vals) in enumerate(series):
            xs = [j + (i - (len(series) - 1) / 2) * w for j in range(5)]
            b = ax.bar(xs, vals, w * 0.95, label=name, color=cols[i % 4])
            ax.bar_label(b, labels=[f'{v:.2f}' for v in vals], fontsize=6, padding=1)
        ax.set_xticks(range(5), CLS + ['macro']); ax.set_ylim(0, 1.08); ax.set_ylabel('F1')
        ax.spines[['top', 'right']].set_visible(False); ax.legend(fontsize=7, frameon=False, ncol=2)
        plt.tight_layout()
        fdir = os.path.join(os.path.dirname(c.out), 'figures')
        os.makedirs(fdir, exist_ok=True)
        fp = os.path.join(fdir, f'fp32_vs_int8_{sp}.png')
        fig.savefig(fp); plt.close(fig)
        out.append(figure(fp, 'Per-class F1 before and after 8-bit quantization.', .85))
    except Exception as e:
        print('Note: FP32/INT8 figure skipped:', e)
    ch = {k: get_metric(m, q, sp, k, 'f1') - get_metric(m, 'edge_fp32', sp, k, 'f1') for k in CLS}
    worst = min(ch, key=ch.get)
    rv = get_metric(m, q, sp, 'V', 'recall') - get_metric(m, 'edge_fp32', sp, 'V', 'recall')
    pv = get_metric(m, q, sp, 'V', 'precision') - get_metric(m, 'edge_fp32', sp, 'V', 'precision')
    tgt = c.targets.get('int8_drop', {}).get('target')
    txt = (f"**What did 8-bit change?** Macro-F1 changed by only {i8 - f32:+.4f}"
           + (f", within the target of at most {tgt} (section 1.4)" if tgt is not None and f32 - i8 <= tgt else (f", more than the target of {tgt}" if tgt is not None else ''))
           + (f"; {100 * (1 - ag):.1f} % of the beats change class" if ag is not None else '') + '. '
           f"The largest change is in class {worst} (F1 {ch[worst]:+.3f}). "
           + (f"The 8-bit model says \"V\" a bit more often: it finds more V beats (recall {rv:+.3f}) but also makes more false V alarms (precision {pv:+.3f}). " if rv > 0 and pv < 0 else '')
           + "So going from 32-bit to 8-bit costs very little; the differences between patients (section 5.2) matter much more. "
           "These numbers come from our PyTorch 8-bit simulation; the official ai8x 8-bit model is compared in section 6.")
    out.append(P(txt))
    return out + own_text(c, '5.3')


def noise_mean(rows, model, snr, col, filt='on', noise=None):
    v = [float(r[col]) for r in rows if r['model'] == model and r.get('filter', 'on') == filt
         and ((snr == 'clean' and r['condition'] == 'clean') or (snr != 'clean' and r['snr_db'] not in ('', None) and float(r['snr_db']) == snr))
         and (noise is None or r['noise_type'] == noise)]
    return sum(v) / len(v) if v else None


def noise_main_table(c):
    """The table the brief asks for (Condition | Accuracy | Macro F1), for the main model, mean of the three noise types."""
    p = os.path.join(c.res, 'noise_results.csv')
    if not os.path.exists(p):
        return []
    rows = csv_rows(p)
    main = 'edge_int8_qat' if any(r['model'] == 'edge_int8_qat' for r in rows) else 'edge_fp32'
    lv = [('Clean', 'clean'), ('12 dB (moderate noise)', 12), ('6 dB (strong noise)', 6), ('0 dB (very strong noise)', 0)]
    t = [['Condition', 'Accuracy', 'Macro F1', 'F1 N', 'F1 S', 'F1 V', 'F1 F']]
    for name, s_ in lv:
        a = noise_mean(rows, main, s_, 'accuracy')
        if a is None:
            continue
        t.append([name, fnum(a), fnum(noise_mean(rows, main, s_, 'macro_f1'))] + [fnum(noise_mean(rows, main, s_, f'f1_{k}'), 3) for k in CLS])
    if len(t) == 1:
        return []
    return [P(f"We add real ECG noise from the MIT-BIH Noise Stress Test Database to the test records: baseline wander (bw), electrode "
              f"motion (em) and muscle artefact (ma). SNR is the signal-to-noise ratio: the lower the number, the stronger the noise. "
              f"The main table shows {NAMES[main]}, averaged over the three noise types."),
            table(t, widths=[W * .28, W * .12, W * .12] + [W * .12] * 4)]


def text_54(c):
    p = os.path.join(c.res, 'noise_results.csv')
    if not os.path.exists(p):
        return own_text(c, '5.4')
    rows = csv_rows(p)
    models = {r['model'] for r in rows}
    main = 'edge_int8_qat' if 'edge_int8_qat' in models else 'edge_fp32'
    mean = lambda *a, **k: noise_mean(rows, *a, **k)
    lv = ['clean', 12, 6, 0]
    mf = [mean(main, s_, 'macro_f1') for s_ in lv]
    if None in mf:
        return own_text(c, '5.4')
    steps = [mf[i] - mf[i + 1] for i in range(3)]
    f1c = {k: mean(main, 'clean', f'f1_{k}') for k in CLS}
    f10 = {k: mean(main, 0, f'f1_{k}') for k in CLS}
    worst = max(CLS, key=lambda k: f1c[k] - f10[k])
    per = {nz: mean(main, 0, 'macro_f1', noise=nz) for nz in ('bw', 'em', 'ma')}
    per = {k: v for k, v in per.items() if v is not None}
    nn = {'bw': 'baseline wander', 'em': 'electrode motion', 'ma': 'muscle artefact'}
    out = [H('What happens when noise is added?'),
           P(f"Macro-F1 goes down from **{mf[0]:.3f}** (clean) to **{mf[1]:.3f}** (12 dB), **{mf[2]:.3f}** (6 dB) and **{mf[3]:.3f}** "
             f"(0 dB). That is a loss of {mf[0] - mf[3]:.3f} in total.")]
    cmp_ = [k for k in ('tiny_cnn_fp32', 'baseline') if k in models and mean(k, 'clean', 'macro_f1') is not None and mean(k, 0, 'macro_f1') is not None]
    if cmp_:
        out.append(B('**Other models:** from clean to 0 dB the macro-F1 drops by ' + ', '.join(
            f"{mean(k, 'clean', 'macro_f1') - mean(k, 0, 'macro_f1'):.3f} for the {NAMES[k]}" for k in cmp_)
            + f", and by {mf[0] - mf[3]:.3f} for our model."))
    out.append(H('Which classes are most affected?'))
    out.append(B(f"**{worst}** loses the most (F1 {f1c[worst]:.2f} -> {f10[worst]:.2f} at 0 dB). Noise changes the width and shape of the "
                 "QRS peak, and that shape is exactly what the model uses to tell V from N."))
    out.append(B('**S and F** change little, but only because they were already very poor without noise.'))
    if per:
        hard, easy = min(per, key=per.get), max(per, key=per.get)
        out.append(B(f"**By noise type:** {nn[hard]} hurts most (macro-F1 {per[hard]:.3f} at 0 dB) and {nn[easy]} hurts least "
                     f"({per[easy]:.3f})."
                     + (' Baseline wander is a slow drift below 0.5 Hz, and our band-pass filter removes it. Electrode motion creates '
                        'sharp jumps in the same frequency range as the heartbeat, so it can look like a beat.'
                        if hard == 'em' and easy == 'bw' else '')))
    out.append(H('Is the drop gradual?'))
    out.append(B(('**Yes.** Each step (clean -> 12 dB -> 6 dB -> 0 dB) costs about the same '
                  f"({', '.join(f'{x:.3f}' for x in steps)}). There is no sudden collapse; the model gets slowly worse as the noise grows.")
                 if max(steps) < 2.5 * max(min(steps), 1e-4) else
                 (f"**No.** The losses per step are {', '.join(f'{x:.3f}' for x in steps)}: the model gets worse faster when the noise is strong.")))
    on = [mean(main, s_, 'macro_f1', 'on') for s_ in lv]
    off = [mean(main, s_, 'macro_f1', 'off') for s_ in lv]
    out.append(H('Does the preprocessing help?'))
    if None not in off:
        gain = sum(a - b for a, b in zip(on, off)) / 4
        every = all(a > b for a, b in zip(on, off))
        out.append(B((f"**Yes.** We ran the same test without the band-pass filter. With the filter, macro-F1 is on average {gain:.3f} higher"
                      + (", at every noise level" if every else '') + ". So the filter must also run on the real device.") if gain > 0 else
                     f"**Not in this run.** Without the band-pass filter macro-F1 changes by {gain:+.3f} on average."))
    else:
        out.append(B('Not tested in this run (run noise_test.py with --ablate-filter).'))
    out.append(H('Could robustness be improved?'))
    out += [B('**Train with noise:** add NSTDB noise during training. The first half of each noise recording is kept for this, so there is '
              'no leakage into the test.'),
            B('**Quality check:** detect very noisy parts (especially electrode motion) and skip them instead of classifying them.'),
            B('**Add timing:** give the model the RR interval (time between beats). Noise changes the shape much more than the timing.')]
    return out + own_text(c, '5.4')

def error_flow(c, text):
    """5.5: the generated error_analysis.md with every figure placed directly after the bullet that describes it."""
    used = set()

    def fig_line(mo):
        name, desc = mo.group(1), mo.group(2)
        if not os.path.exists(os.path.join(c.res, name)):
            return mo.group(0)
        used.add(name)
        return f"![{desc[0].upper() + desc[1:]}.]({name}|0.95)"
    text = re.sub(r'^\* `(error_[^`]+\.png)`:\s*(.+)$', fig_line, text, flags=re.M)
    fl = [P("The tables below come from `error_analysis.py` on the final model. Each example figure shows six misclassified beats: the top "
            "row is the 1-second model input, the bottom row the filtered ECG around it (the shaded part is the beat). Half of the examples are "
            "the most confident errors and half are chosen at random.")]
    fl += md_block(c, text, shift=1)
    for p_ in sorted(glob.glob(os.path.join(c.res, 'error_*.png'))):
        if os.path.basename(p_) not in used:
            fl.append(figure(p_, os.path.basename(p_), .9))
    return fl


def evaluation_tables(c, text):
    """Render tables_<split>.md section by section and put the explanation after each part (5.1, 5.2, 5.3)."""
    text = re.sub(r'\A# .*\n', '', text)
    text = text.replace('## 5.1 Overall results', '## 5.1 Overall Results').replace('## 5.2 Per-class performance', '## 5.2 Per-Class Performance')
    parts = re.split(r'(?m)^(?=## )', text)
    fl = []
    for part in parts:
        if not part.strip():
            continue
        head = part.split('\n', 1)[0]
        if head.startswith('## 5.3'):
            fl.append(Paragraph('5.3 Quantization Analysis', ST['h2']))
            fl += quant_compare(c)
            continue
        fl += md_block(c, part, shift=0)
        if head.startswith('## 5.1'):
            fl += text_51(c)
        elif head.startswith('## 5.2'):
            for name, cap in [(f'confusion_matrix_edge_int8_qat_{c.split}.png', 'Confusion matrix, INT8 model.'),
                              (f'confusion_matrix_edge_fp32_{c.split}.png', 'Confusion matrix, FP32 model.'),
                              (f'confusion_matrix_baseline_{c.split}.png', 'Confusion matrix, classical baseline.')]:
                p_ = os.path.join(c.res, name)
                if os.path.exists(p_):
                    fl.append(figure(p_, cap, .6))
            fl += text_52(c)
    return fl


def task2_flow(c):
    n, fl = c.notes, []
    fl.append(PageBreak())
    fl.append(Paragraph('4. Modeling', ST['h1']))
    fl += baseline_section(c) + architecture_section(c) + text_42(c)
    tc = training_config_section(c)
    fl += tc[:1] + text_43(c) + tc[1:]
    sel = selection_section(c)
    fl += sel[:1] + text_44(c) + sel[1:]
    p = os.path.join(c.res, 'training_curves_edge_cnn.png')
    if os.path.exists(p):
        fl.append(figure(p, 'Training curves of the final edge model (validation records).', .85))
    fl += quantization_section(c)
    fl.append(Paragraph('5. Evaluation', ST['h1']))
    tp = rp(c, f'tables_{c.split}.md')
    if tp:
        fl += evaluation_tables(c, read(tp))
    fl.append(Paragraph('5.4 Noise Robustness', ST['h2']))
    fl += noise_main_table(c)
    for name, cap in [('noise_robustness.png', 'Accuracy and macro-F1 versus noise level (SNR).'), ('noise_per_class_f1.png', 'F1 of each class versus noise level.'),
                      ('noise_filter_ablation.png', 'With and without the band-pass filter.')]:
        p = os.path.join(c.res, name)
        if os.path.exists(p):
            fl.append(figure(p, cap, .8))
    fl += text_54(c)
    np_ = rp(c, 'noise_results_table.md')
    if np_:
        fl.append(H('More detail: all models and each noise type'))
        fl += md_block(c, read(np_), shift=2)
    fl.append(Paragraph('5.5 Error Analysis', ST['h2']))
    ep = rp(c, 'error_analysis.md')
    if ep:
        fl += error_flow(c, read(ep).split('## 6.')[0])
    if n.get('error_discussion'):
        fl += [Paragraph(inline(x), ST['p']) for x in str(n['error_discussion']).strip().split('\n\n')]
    fl += own_text(c, '5.5')
    fl += criteria_section(c)
    fl += deployment_section(c)
    fl += final_discussion(c)
    fl += limitations_section(c)
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


def build(c, formats=('pdf', 'docx')):
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
    written = []
    if 'docx' in formats:                     # written first: reportlab may split the parts while it builds the PDF
        from docx_writer import write_docx
        skipped = write_docx(flow, c.docx_out, title=n.get('title', 'Full report'), author=n.get('authors', ''))
        if skipped:
            print('Note: parts not written to Word:', ', '.join(skipped))
        written.append(c.docx_out)
    if 'pdf' not in formats:
        return written
    doc = SimpleDocTemplate(c.out, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title=n.get('title', 'Full report'), author=n.get('authors', ''))

    def footer(canvas, d):
        canvas.setFont(SANS, 7); canvas.setFillColor(colors.grey)
        if d.page > 1:
            canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f'Page {d.page}')
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    return written + [c.out]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='../configs/best_config.yaml')
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--notes'); ap.add_argument('--task1'); ap.add_argument('--results-dir'); ap.add_argument('--max78002-dir'); ap.add_argument('--out')
    ap.add_argument('--format', default='both', choices=['pdf', 'docx', 'both'], help='which files to write (default: both)')
    ap.add_argument('--docx-out', help='Word file (default: same name as the PDF, ending .docx)')
    a = ap.parse_args()
    c = load(a)
    out = build(c, ('pdf', 'docx') if a.format == 'both' else (a.format,))
    print('Report written to', ', '.join(out))
    if c.missing:
        print('Missing inputs (parts skipped):', ', '.join(sorted(set(c.missing))))


if __name__ == '__main__':
    main()
