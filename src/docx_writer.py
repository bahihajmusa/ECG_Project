"""Write the report as a Word document (.docx) from the SAME content that make_report.py builds for the PDF.

make_report.py builds the report as a list of reportlab parts (paragraphs, tables, figures, spacers, page breaks).
write_docx() walks that list and writes each part with python-docx, so the PDF and the Word file always contain
exactly the same text, numbers, tables and figures. The Word file can then be edited by hand if needed.

Needs: pip install python-docx
Used by make_report.py (--format docx or --format both); it can not be run on its own.
"""
import html
import re

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, Spacer, Table

# Fonts of the Word file (the PDF uses DejaVu Serif / Sans). Cambria and Calibri exist in every Word installation.
BODY_FONT, TABLE_FONT, MONO_FONT = 'Cambria', 'Calibri', 'Consolas'
GREY = RGBColor(0x80, 0x80, 0x80)
HEADER_FILL = 'E8EEF6'                       # same light blue as the PDF table header
GRID_COLOR = 'D3D3D3'                        # same light grey grid as the PDF
PT_PER_MM = 72 / 25.4

# reportlab style name -> how the paragraph is written in Word.
# kind: heading level, or None for a normal paragraph; size in pt; align: 0 left, 1 center, 4 justified
STYLE_MAP = {
    't':       dict(level=None, size=24, bold=True, align=1, space_after=10, title=True),
    'c':       dict(level=None, size=12, align=1, space_after=6),
    'h1':      dict(level=1, size=16, align=1, space_before=10, space_after=8),
    'h2':      dict(level=2, size=13, align=0, space_before=9, space_after=5),
    'h3':      dict(level=3, size=10.5, align=0, space_before=6, space_after=3),
    'h3c':     dict(level=3, size=10.5, align=1, space_before=6, space_after=3),
    'p':       dict(level=None, size=10, align=4, space_after=5),
    'b':       dict(level=None, size=10, align=4, space_after=2, bullet=True),
    'figcap':  dict(level=None, size=7.5, align=1, space_after=6, grey=True, font=TABLE_FONT),
    'cap':     dict(level=None, size=7.5, align=0, space_after=6, grey=True, font=TABLE_FONT),
    'cell':    dict(level=None, size=8, align=0, space_after=0, font=TABLE_FONT),
    'cellb':   dict(level=None, size=8, align=0, space_after=0, font=TABLE_FONT, bold=True),
}
ALIGN = {0: WD_ALIGN_PARAGRAPH.LEFT, 1: WD_ALIGN_PARAGRAPH.CENTER, 2: WD_ALIGN_PARAGRAPH.RIGHT, 4: WD_ALIGN_PARAGRAPH.JUSTIFY}
TOKEN = re.compile(r'(<br\s*/?>|</?b>|</?i>|</?u>|<font[^>]*>|</font>)', re.I)


# ---------------------------------------------------------------- inline markup (<b>, <i>, <font face=Mono>, <br/>)
def add_markup(par, markup, size, bold=False, font=None, color=None):
    """Write reportlab paragraph markup as Word runs (bold, italic, monospace, line breaks)."""
    state = {'b': bold, 'i': False, 'u': False}
    fonts = [(font, size)]                  # stack for <font ...> ... </font>
    for tok in TOKEN.split(markup or ''):
        if not tok:
            continue
        low = tok.lower()
        if re.fullmatch(r'<br\s*/?>', low):
            par.add_run().add_break()
        elif low in ('<b>', '<i>', '<u>'):
            state[low[1]] = True
        elif low in ('</b>', '</i>', '</u>'):
            state[low[2]] = bold if low[2] == 'b' else False
        elif low.startswith('<font'):
            face = re.search(r'face="([^"]+)"', tok)
            sz = re.search(r'size="([\d.]+)"', tok)
            f = MONO_FONT if face and 'mono' in face.group(1).lower() else fonts[-1][0]
            fonts.append((f, float(sz.group(1)) if sz else fonts[-1][1]))
        elif low == '</font>':
            if len(fonts) > 1:
                fonts.pop()
        else:
            r = par.add_run(html.unescape(tok).replace(' ', ' '))
            r.bold, r.italic, r.underline = state['b'], state['i'], state['u']
            f, s = fonts[-1]
            r.font.size = Pt(s)
            if f:
                r.font.name = f
                r.element.rPr.rFonts.set(qn('w:eastAsia'), f)
            if color is not None:
                r.font.color.rgb = color


def style_of(p):
    return STYLE_MAP.get(p.style.name, STYLE_MAP['p'])


def write_paragraph(doc_or_cell, p, keep_with_next=False):
    """One reportlab Paragraph -> one Word paragraph (heading, bullet, caption or body text)."""
    s = style_of(p)
    if s.get('level'):
        par = doc_or_cell.add_heading(level=s['level'])
    elif s.get('bullet') or getattr(p, 'bulletText', None):
        par = doc_or_cell.add_paragraph(style='List Bullet')
    else:
        par = doc_or_cell.add_paragraph()
    fmt = par.paragraph_format
    fmt.space_before = Pt(s.get('space_before', 0))
    fmt.space_after = Pt(s.get('space_after', 4))
    fmt.alignment = ALIGN.get(s.get('align', p.style.alignment), WD_ALIGN_PARAGRAPH.LEFT)
    if 'face="mono' in (p.text or '').lower() and '<br' in (p.text or '').lower():
        fmt.alignment = WD_ALIGN_PARAGRAPH.LEFT          # log listings: never stretch the lines
    if s.get('level') or keep_with_next:
        fmt.keep_with_next = True
    add_markup(par, p.text, s['size'], bold=s.get('bold', bool(s.get('level'))), font=s.get('font'),
               color=GREY if s.get('grey') else None)
    return par


# ---------------------------------------------------------------- tables
def set_cell_fill(cell, hex_fill):
    tc = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), hex_fill)
    tc.append(shd)


def _insert_tblpr(tblPr, el):
    """Put an element into w:tblPr at the position the Word file format requires."""
    order = ['tblStyle', 'tblpPr', 'tblOverlap', 'bidiVisual', 'tblStyleRowBandSize', 'tblStyleColBandSize', 'tblW', 'jc',
             'tblCellSpacing', 'tblInd', 'tblBorders', 'shd', 'tblLayout', 'tblCellMar', 'tblLook', 'tblCaption', 'tblDescription']
    name = el.tag.split('}')[1]
    later = order[order.index(name) + 1:]
    for child in tblPr:
        if child.tag.split('}')[1] in later:
            child.addprevious(el)
            return
    tblPr.append(el)


def _fix_zoom(doc):
    """python-docx's template has <w:zoom> without the required percent attribute; Word's validator flags it."""
    z = doc.settings.element.find(qn('w:zoom'))
    if z is not None and z.get(qn('w:percent')) is None:
        z.set(qn('w:percent'), '100')


def set_table_borders(table, color=GRID_COLOR, size=4):
    tblPr = table._tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement(f'w:{edge}')
        el.set(qn('w:val'), 'single'); el.set(qn('w:sz'), str(size)); el.set(qn('w:space'), '0'); el.set(qn('w:color'), color)
        borders.append(el)
    _insert_tblpr(tblPr, borders)


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement('w:tblHeader'); el.set(qn('w:val'), 'true')
    trPr.append(el)


def write_table(doc, t):
    """reportlab Table -> Word table with the same column widths, light grid and shaded header row."""
    rows = t._cellvalues
    if not rows:
        return
    ncol = max(len(r) for r in rows)
    widths = [w if isinstance(w, (int, float)) else None for w in (t._colWidths or [None] * ncol)]
    table = doc.add_table(rows=len(rows), cols=ncol)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    header = all(isinstance(c, Paragraph) and c.style.name == 'cellb' for c in rows[0])
    for j, w in enumerate(widths):
        if w:
            table.columns[j].width = Pt(w)
    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = table.cell(i, j)
            if widths[j]:
                cell.width = Pt(widths[j])
            val = row[j] if j < len(row) else ''
            first = cell.paragraphs[0]
            if isinstance(val, Paragraph):
                s = style_of(val)
                add_markup(first, val.text, s['size'], bold=s.get('bold', False), font=s.get('font', TABLE_FONT))
            else:
                add_markup(first, html.escape(str(val)), 8, font=TABLE_FONT)
            first.paragraph_format.space_after = Pt(0)
            if header and i == 0:
                set_cell_fill(cell, HEADER_FILL)
    if header:
        repeat_header(table.rows[0])
    gap = doc.add_paragraph()
    gap.paragraph_format.space_after = Pt(2)


# ---------------------------------------------------------------- figures, spacers, page breaks
def write_image(doc, im, keep_with_next=False):
    doc.add_picture(im.filename, width=Pt(im.drawWidth))
    par = doc.paragraphs[-1]
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.keep_with_next = keep_with_next
    par.paragraph_format.space_after = Pt(2)


def write_spacer(doc, sp):
    par = doc.add_paragraph()
    par.paragraph_format.space_after = Pt(max(sp.height - 6, 0))
    par.paragraph_format.line_spacing = Pt(6)


def page_break(doc):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def write_flow(doc, flow, warnings):
    for k, item in enumerate(flow):
        nxt = flow[k + 1] if k + 1 < len(flow) else None
        if isinstance(item, KeepTogether):
            write_flow(doc, list(item._content), warnings)
        elif isinstance(item, Paragraph):
            write_paragraph(doc, item, keep_with_next=isinstance(nxt, (Image, Table)))
        elif isinstance(item, Table):
            write_table(doc, item)
        elif isinstance(item, Image):
            write_image(doc, item, keep_with_next=isinstance(nxt, Paragraph) and nxt.style.name == 'figcap')
        elif isinstance(item, PageBreak):
            page_break(doc)
        elif isinstance(item, Spacer):
            write_spacer(doc, item)
        else:
            warnings.append(type(item).__name__)


# ---------------------------------------------------------------- document setup
def add_page_number_footer(section):
    """'Page N' at the bottom right of every page except the title page (as in the PDF)."""
    section.different_first_page_header_footer = True
    par = section.footer.paragraphs[0]
    par.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = par.add_run('Page ')
    r.font.size, r.font.color.rgb = Pt(7), GREY
    run = par.add_run()
    run.font.size, run.font.color.rgb = Pt(7), GREY
    for tag, txt in (('begin', None), (None, 'PAGE'), ('end', None)):
        if tag:
            el = OxmlElement('w:fldChar'); el.set(qn('w:fldCharType'), tag)
        else:
            el = OxmlElement('w:instrText'); el.set(qn('xml:space'), 'preserve'); el.text = txt
        run._r.append(el)


def setup_document(title, author):
    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.PORTRAIT
    sec.page_width, sec.page_height = Mm(210), Mm(297)          # A4, margins as in the PDF
    sec.left_margin = sec.right_margin = Mm(18)
    sec.top_margin = sec.bottom_margin = Mm(16)
    sec.footer_distance = Mm(7)
    normal = doc.styles['Normal']
    normal.font.name, normal.font.size = BODY_FONT, Pt(10)
    normal.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_FONT)
    normal.paragraph_format.line_spacing = 1.08
    for lvl, size in ((1, 16), (2, 13), (3, 10.5)):
        h = doc.styles[f'Heading {lvl}']
        h.font.name, h.font.size, h.font.bold = BODY_FONT, Pt(size), True
        h.font.color.rgb = RGBColor(0, 0, 0)
        h.font.italic = False
        rpr = h.element.get_or_add_rPr()
        rf = rpr.find(qn('w:rFonts'))
        if rf is None:
            rf = OxmlElement('w:rFonts'); rpr.append(rf)
        for a in ('w:ascii', 'w:hAnsi', 'w:cs', 'w:eastAsia'):
            rf.set(qn(a), BODY_FONT)
        for a in ('w:asciiTheme', 'w:hAnsiTheme', 'w:cstheme', 'w:eastAsiaTheme'):
            rf.attrib.pop(qn(a), None)
        h.paragraph_format.keep_with_next = True
    doc.core_properties.title = title or ''
    doc.core_properties.author = author or ''
    add_page_number_footer(sec)
    return doc


def write_docx(flow, path, title='', author=''):
    """Write the reportlab flow to `path` as .docx. Returns a list of part types that were skipped (normally empty)."""
    doc = setup_document(title, author)
    warnings = []
    write_flow(doc, flow, warnings)
    _fix_zoom(doc)
    doc.save(path)
    return sorted(set(warnings))
