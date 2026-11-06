"""Extract the figures that are saved inside notebooks/EDA.ipynb (as cell outputs) into results/eda/.

Run it after you have run the notebook and saved it WITH outputs:
    python export_eda_figures.py            (from src/)
make_report.py then uses these files, so the report always shows the figures of the current notebook.
Two extra figures (split composition, leakage experiment) are drawn from the numbers the notebook printed.
"""
import argparse, base64, json, os, re

# (text that appears in the code cell, names of the figures that cell produces, in order)
CELL_FIGURES = [
    ('def plot_ecg_signal', ['ecg_signal']),
    ('Both channels of one record', ['channels']),
    ('counts = df_beats', ['class_distribution']),
    ('rec_class = df_beats.groupby', ['record_distribution']),
    ('HALF_B, HALF_A', ['example_beats']),
    ('Average beat per class', ['average_beat']),
    ('Amplitude distribution of the selected lead', ['amplitude']),
    ('Differences between RECORDINGS', ['recording_differences']),
    ('Differences between CHANNELS', ['channel_differences']),
    ('Noise level per 2-second window', ['noise_per_record', 'noise_windows']),
    ('NOISE = {}', ['nstdb_records']),
    ('A clean DS1 ECG with NSTDB noise', ['noise_snr_examples']),
    ('Preprocessing illustration', ['preprocessing']),
]


def cell_text(cell):
    out = []
    for o in cell.get('outputs', []):
        if o.get('output_type') == 'stream':
            out.append(''.join(o.get('text', [])))
        elif 'data' in o and 'text/plain' in o['data']:
            out.append(''.join(o['data']['text/plain']))
    return '\n'.join(out)


def extra_figures(nb, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    text = '\n'.join(cell_text(c) for c in nb['cells'] if c['cell_type'] == 'code')
    made = []
    m1 = re.search(r'random beats \(leaky\)\s+([\d.]+)\s+([\d.]+)', text)
    m2 = re.search(r'by record \(correct\)\s+([\d.]+)\s+([\d.]+)', text)
    if m1 and m2:
        vals = [float(m1.group(2)), float(m2.group(2))]
        fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=200)
        b = ax.bar(['Random beat split\n(leakage)', 'Split by record\n(our protocol)'], vals, color=['#b0b0b0', '#1f4e79'], width=.55)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + .02, f'{v:.2f}', ha='center', fontsize=10)
        ax.set_ylim(0, 1.05); ax.set_ylabel('Macro-F1'); ax.set_title('Same Random Forest, two ways of splitting', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False); plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'leakage_demo.png')); plt.close(); made.append('leakage_demo')
    rows = re.findall(r'^\s*\d+\s+(Train|Validation|Test)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$', text, re.M)
    if len(rows) >= 3:
        C = {'N': '#2ecc71', 'S': '#3498db', 'V': '#e74c3c', 'F': '#f39c12'}
        labels = {'Train': 'Training', 'Validation': 'Validation', 'Test': 'Test = DS2'}
        names = [f"{labels[r[0]]}\n({r[1]} records)" for r in rows[:3]]
        cnt = {k: [int(r[2 + i]) for r in rows[:3]] for i, k in enumerate('NSVF')}
        tot = [int(r[6]) for r in rows[:3]]
        fig, ax = plt.subplots(figsize=(7, 2.8), dpi=200)
        left = [0, 0, 0]
        for k in 'NSVF':
            w = [cnt[k][i] / tot[i] * 100 for i in range(3)]
            ax.barh(names, w, left=left, color=C[k], label=k, edgecolor='white')
            left = [left[i] + w[i] for i in range(3)]
        ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel('Share of beats in the split (%)')
        for i in range(3):
            ax.text(101, i, f'{tot[i]:,} beats', va='center', fontsize=8)
        ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.22), frameon=False, fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        plt.subplots_adjust(right=.82, left=.2, bottom=.22, top=.85)
        plt.savefig(os.path.join(outdir, 'split_composition.png')); plt.close(); made.append('split_composition')
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--notebook', default='../notebooks/EDA.ipynb')
    ap.add_argument('--out', default='../results/eda')
    a = ap.parse_args()
    nb = json.load(open(a.notebook, encoding='utf-8'))
    os.makedirs(a.out, exist_ok=True)
    done = []
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        pngs = [o['data']['image/png'] for o in cell.get('outputs', []) if 'data' in o and 'image/png' in o['data']]
        if not pngs:
            continue
        for key, names in CELL_FIGURES:
            if key in src:
                for n, p in zip(names, pngs):
                    with open(os.path.join(a.out, f'eda_{n}.png'), 'wb') as f:
                        f.write(base64.b64decode(p))
                    done.append(n)
                break
    done += extra_figures(nb, a.out)
    print('Wrote', len(done), 'figures to', a.out, ':', ', '.join(done))
    missing = [n for _, ns in CELL_FIGURES for n in ns if n not in done]
    if missing:
        print('Not found in the notebook outputs (run all cells and save with outputs):', ', '.join(missing))


if __name__ == '__main__':
    main()
