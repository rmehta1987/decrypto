#!/usr/bin/env python
"""Build the SC26 talk "Can LLMs reason?" as an 8-slide PowerPoint in UChicago style.

Usage:
    python slides/build_sc26_deck.py [--out slides/can_llms_reason_sc26.pptx] [--preview DIR]

Needs python-pptx. --preview also needs matplotlib and Pillow; it draws each slide from
the same layout calls, measuring text with DejaVu Condensed fonts (as wide as or wider
than Arial and Garamond), and prints a warning for any text box that overflows.

Every number on the slides is sourced in that slide's speaker notes.
"""
import argparse
import os
import re

from pptx import Presentation
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from lxml import etree

W, H = 13.333, 7.5
N_SLIDES = 8

# UChicago brand chrome (titles, frame, verdict bar).
MAROON = "800000"
# Chart marks: nearest shades of the brand maroon / lake blue / goldenrod that pass the
# dataviz validator on white, all pairs (lightness band, chroma, CVD, contrast).
DEC = "A4343A"    # Decrypto
MAR = "2672AE"    # MARSHAL
GOLD = "B88300"   # MARSHAL game-playing engines
# Ink and greys (UChicago greystones).
INK = "1A1A1A"
INK2 = "4D4D4A"
MUTED = "767676"
GREY_L = "D6D6CE"
GREY_M = "B4B4AC"
GREY_D = "8A8A84"
TINT = "F4F4F0"
WHITE = "FFFFFF"

SERIF = "Garamond"
SANS = "Arial"
FONT_DIR = "/usr/share/fonts/truetype"
MEASURE_FONTS = {
    ("sans", False, False): "DejaVuSansCondensed.ttf",
    ("sans", True, False): "DejaVuSansCondensed-Bold.ttf",
    ("sans", False, True): "DejaVuSansCondensed-Oblique.ttf",
    ("sans", True, True): "DejaVuSansCondensed-BoldOblique.ttf",
    ("serif", False, False): "DejaVuSerifCondensed.ttf",
    ("serif", True, False): "DejaVuSerifCondensed-Bold.ttf",
    ("serif", False, True): "DejaVuSerifCondensed-Italic.ttf",
    ("serif", True, True): "DejaVuSerifCondensed-BoldItalic.ttf",
}


def rgb(h):
    return RGBColor.from_string(h)


# --------------------------------------------------------------------------- text model
def P(text, size=14, family="sans", color=INK, bold=False, italic=False, align="l",
      sb=0, sa=0, ls=1.0, spc=0, hang=None):
    """A paragraph. **text** marks bold runs. hang=(left_margin_in, hang_in) for lists."""
    runs = []
    for i, piece in enumerate(re.split(r"\*\*", text)):
        if piece:
            runs.append((piece, {"bold": bold or (i % 2 == 1), "italic": italic, "color": color}))
    return {"runs": runs, "size": size, "family": family, "align": align, "sb": sb, "sa": sa,
            "ls": ls, "spc": spc, "hang": hang}


class Measurer:
    def __init__(self):
        from PIL import ImageFont
        self._IF = ImageFont
        self._cache = {}

    def font(self, family, bold, italic, size):
        key = (family, bold, italic, size)
        if key not in self._cache:
            path = os.path.join(FONT_DIR, MEASURE_FONTS[(family, bold, italic)])
            self._cache[key] = self._IF.truetype(path, size=int(round(size * 20)))
        return self._cache[key]

    def width(self, text, family, bold, italic, size, spc=0):
        f = self.font(family, bold, italic, size)
        return f.getlength(text) / 20.0 + len(text) * spc / 100.0


def wrap_para(m, p, width_pt):
    """Greedy word wrap. Returns ([(tokens, width)], too_long_words)."""
    toks = []
    for text, st in p["runs"]:
        for piece in re.split(r"( |\t)", text):
            if piece:
                if piece == "\t":
                    toks.append((piece, st, None))
                    continue
                w = m.width(piece, p["family"], st["bold"], st["italic"], p["size"], p["spc"])
                toks.append((piece, st, w))
    lines, cur, curw, too_long = [], [], 0.0, []

    def flush():
        nonlocal cur, curw
        while cur and cur[-1][0] == " ":
            curw -= cur[-1][2]
            cur.pop()
        lines.append((cur, curw))
        cur, curw = [], 0.0

    for t in toks:
        if t[0] == "\t":
            hang = p["hang"][1] * 72 if p["hang"] else 0.0
            t = (t[0], t[1], max(hang - curw, 2.0))
            cur.append(t)
            curw += t[2]
            continue
        if t[0] == " ":
            if cur:
                cur.append(t)
                curw += t[2]
            continue
        if t[2] > width_pt:
            too_long.append(t[0])
        if cur and curw + t[2] > width_pt:
            flush()
        cur.append(t)
        curw += t[2]
    flush()
    return lines, too_long


def line_height(p):
    return p["size"] * 1.2 * p["ls"]


# --------------------------------------------------------------------------- deck
class Deck:
    def __init__(self, preview):
        self.prs = Presentation()
        self.prs.slide_width = Inches(W)
        self.prs.slide_height = Inches(H)
        self.blank = self.prs.slide_layouts[6]
        self.pages = []
        self.warnings = []
        self.m = Measurer() if preview else None
        self.n = 0

    # ---- slide
    def slide(self, notes=""):
        self.s = self.prs.slides.add_slide(self.blank)
        self.ops = []
        self.pages.append(self.ops)
        self.n += 1
        if notes:
            self.s.notes_slide.notes_text_frame.text = notes.strip()
        return self.s

    # ---- shapes
    def rect(self, x, y, w, h, fill=None, line=None, lw=0.75, radius=None, oval=False):
        kind = MSO_SHAPE.OVAL if oval else (MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE)
        shp = self.s.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
        if radius and not oval:
            shp.adjustments[0] = min(0.5, radius / min(w, h))
        if fill:
            shp.fill.solid()
            shp.fill.fore_color.rgb = rgb(fill)
        else:
            shp.fill.background()
        if line:
            shp.line.color.rgb = rgb(line)
            shp.line.width = Pt(lw)
        else:
            shp.line.fill.background()
        shp.shadow.inherit = False
        self.ops.append(("rect", dict(x=x, y=y, w=w, h=h, fill=fill, line=line, lw=lw,
                                      radius=radius, oval=oval)))
        return shp

    def line(self, x1, y1, x2, y2, color=GREY_L, lw=1.0, arrow=False):
        c = self.s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                        Inches(x2), Inches(y2))
        c.line.color.rgb = rgb(color)
        c.line.width = Pt(lw)
        if arrow:
            ln = c.line._get_or_add_ln()
            te = etree.SubElement(ln, qn("a:tailEnd"))
            te.set("type", "triangle")
            te.set("w", "med")
            te.set("len", "med")
        self.ops.append(("line", dict(x1=x1, y1=y1, x2=x2, y2=y2, color=color, lw=lw, arrow=arrow)))
        return c

    def text(self, x, y, w, h, paras, anchor="t", fill=None, pad=0.0, name=""):
        if isinstance(paras, dict):
            paras = [paras]
        if fill:
            self.rect(x, y, w, h, fill=fill)
        tb = self.s.shapes.add_textbox(Inches(x + pad), Inches(y + pad), Inches(w - 2 * pad),
                                       Inches(h - 2 * pad))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
        for i, p in enumerate(paras):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[p["align"]]
            para.space_before = Pt(p["sb"])
            para.space_after = Pt(p["sa"])
            para.line_spacing = p["ls"]
            if p["hang"]:
                left, hang = p["hang"]
                pPr = para._p.get_or_add_pPr()
                pPr.set("marL", str(int(Emu(Inches(left)))))
                pPr.set("indent", str(-int(Emu(Inches(hang)))))
            for txt, st in p["runs"]:
                r = para.add_run()
                r.text = txt
                f = r.font
                f.name = SERIF if p["family"] == "serif" else SANS
                f.size = Pt(p["size"])
                f.bold = st["bold"]
                f.italic = st["italic"]
                f.color.rgb = rgb(st["color"])
                if p["spc"]:
                    r._r.get_or_add_rPr().set("spc", str(int(p["spc"])))
        self.ops.append(("text", dict(x=x + pad, y=y + pad, w=w - 2 * pad, h=h - 2 * pad,
                                      paras=paras, anchor=anchor, name=name)))
        if self.m:
            self._check(x + pad, y + pad, w - 2 * pad, h - 2 * pad, paras, name)
        return tb

    def _check(self, x, y, w, h, paras, name):
        total = 0.0
        for i, p in enumerate(paras):
            indent = p["hang"][0] if p["hang"] else 0.0
            lines, too_long = wrap_para(self.m, p, (w - indent) * 72)
            total += p["sb"] + len(lines) * line_height(p) + (p["sa"] if i < len(paras) - 1 else 0)
            for t in too_long:
                self.warnings.append(f"slide {self.n}: word wider than box: {t!r}")
        if total > h * 72 + 1.5:
            first = "".join(t for t, _ in paras[0]["runs"])[:60]
            self.warnings.append(
                f"slide {self.n}: text overflows by {(total - h * 72) / 72:.2f} in "
                f"(needs {total / 72:.2f} in, box {h:.2f} in): {name or first!r}")

    # ---- reusable pieces
    def frame(self, kicker, title, marker=None):
        self.rect(0, 0, W, 0.12, fill=MAROON)
        kx = 0.6
        if marker:
            self.rect(0.6, 0.445, 0.15, 0.15, fill=marker)
            kx = 0.86
        self.text(kx, 0.40, 9.0, 0.25, P(kicker, size=11, bold=True, color=INK2, spc=150))
        self.text(0.6, 0.68, 12.13, 0.98, P(title, size=28, family="serif", color=MAROON, ls=0.95),
                  name="title")
        self.line(0.6, 1.72, 12.73, 1.72, color=GREY_L, lw=1.0)
        self.line(0.6, 7.0, 12.73, 7.0, color=GREY_L, lw=0.75)
        self.text(0.6, 7.08, 4.5, 0.25, P("THE UNIVERSITY OF CHICAGO", size=9, bold=True,
                                          color=MAROON, spc=200))
        self.text(4.67, 7.08, 4.0, 0.25, P("Can LLMs reason?  ·  SC26", size=9, color=MUTED, align="c"))
        self.text(8.73, 7.08, 4.0, 0.25, P(f"{self.n} / {N_SLIDES}", size=9, color=MUTED, align="r"))

    def therefore(self, x, y, w, h, text, size=14):
        self.rect(x, y, 0.07, h, fill=MAROON)
        self.text(x + 0.22, y, w - 0.22, h, P(text, size=size, color=INK), anchor="m", name="therefore")

    def gpu_row(self, x, y, fills, size=0.3, gap=0.07, idle_line=GREY_M):
        for i, f in enumerate(fills):
            if f:
                self.rect(x + i * (size + gap), y, size, size, fill=f, radius=0.04)
            else:
                self.rect(x + i * (size + gap), y, size, size, line=idle_line, lw=1.0, radius=0.04)

    # ---- output
    def save(self, path):
        self.prs.save(path)

    def render(self, outdir):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
        from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse

        os.makedirs(outdir, exist_ok=True)
        fps = {}

        def fp(family, bold, italic):
            k = (family, bold, italic)
            if k not in fps:
                fps[k] = FontProperties(fname=os.path.join(FONT_DIR, MEASURE_FONTS[k]))
            return fps[k]

        paths = []
        for idx, ops in enumerate(self.pages, 1):
            fig = plt.figure(figsize=(W, H), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.axis("off")
            ax.add_patch(Rectangle((0, 0), W, H, color="white", zorder=0))
            z = 1
            for kind, o in ops:
                z += 1
                if kind == "rect":
                    fc = "#" + o["fill"] if o["fill"] else "none"
                    ec = "#" + o["line"] if o["line"] else "none"
                    lw = o["lw"] if o["line"] else 0
                    if o["oval"]:
                        pat = Ellipse((o["x"] + o["w"] / 2, o["y"] + o["h"] / 2), o["w"], o["h"],
                                      fc=fc, ec=ec, lw=lw, zorder=z)
                    elif o["radius"]:
                        r = min(o["radius"], o["w"] / 2, o["h"] / 2)
                        pat = FancyBboxPatch((o["x"] + r, o["y"] + r), o["w"] - 2 * r, o["h"] - 2 * r,
                                             boxstyle=f"round,pad={r}", fc=fc, ec=ec, lw=lw, zorder=z)
                    else:
                        pat = Rectangle((o["x"], o["y"]), o["w"], o["h"], fc=fc, ec=ec, lw=lw, zorder=z)
                    ax.add_patch(pat)
                elif kind == "line":
                    if o["arrow"]:
                        ax.annotate("", xy=(o["x2"], o["y2"]), xytext=(o["x1"], o["y1"]), zorder=z,
                                    arrowprops=dict(arrowstyle="-|>", color="#" + o["color"],
                                                    lw=o["lw"], shrinkA=0, shrinkB=0,
                                                    mutation_scale=12))
                    else:
                        ax.plot([o["x1"], o["x2"]], [o["y1"], o["y2"]], color="#" + o["color"],
                                lw=o["lw"], zorder=z, solid_capstyle="butt")
                else:
                    self._render_text(ax, o, fp, z)
            p = os.path.join(outdir, f"slide_{idx}.png")
            fig.savefig(p, dpi=100)
            plt.close(fig)
            paths.append(p)
        return paths

    def _render_text(self, ax, o, fp, z):
        x, y, w, h, paras = o["x"], o["y"], o["w"], o["h"], o["paras"]
        laid = []
        total = 0.0
        for i, p in enumerate(paras):
            indent = p["hang"][0] if p["hang"] else 0.0
            lines, _ = wrap_para(self.m, p, (w - indent) * 72)
            laid.append((p, lines, indent))
            total += p["sb"] + len(lines) * line_height(p) + (p["sa"] if i < len(paras) - 1 else 0)
        off = {"t": 0.0, "m": (h * 72 - total) / 2, "b": h * 72 - total}[o["anchor"]]
        cy = y * 72 + off
        for i, (p, lines, indent) in enumerate(laid):
            cy += p["sb"]
            for j, (toks, lw) in enumerate(lines):
                lh = line_height(p)
                base = cy + lh * 0.8
                left = indent - (p["hang"][1] if (p["hang"] and j == 0) else 0.0)
                if p["align"] == "l":
                    cx = (x + left) * 72
                elif p["align"] == "c":
                    cx = (x + w / 2) * 72 - lw / 2
                else:
                    cx = (x + w) * 72 - lw
                for t, st, tw in toks:
                    if t == "\t":
                        cx += tw
                        continue
                    f = fp(p["family"], st["bold"], st["italic"])
                    if p["spc"]:
                        for ch in t:
                            ax.text(cx / 72, base / 72, ch, fontproperties=f, fontsize=p["size"],
                                    color="#" + st["color"], va="baseline", ha="left", zorder=z)
                            cx += self.m.width(ch, p["family"], st["bold"], st["italic"], p["size"],
                                               p["spc"])
                        continue
                    ax.text(cx / 72, base / 72, t, fontproperties=f, fontsize=p["size"],
                            color="#" + st["color"], va="baseline", ha="left", zorder=z)
                    cx += tw
                cy += lh
            cy += p["sa"]


# --------------------------------------------------------------------------- slides
def slide1(d):
    d.slide(notes=NOTES[1])
    d.rect(0, 0, W, H, fill=MAROON)
    d.text(0.9, 0.75, 8, 0.3, P("THE UNIVERSITY OF CHICAGO", size=12, bold=True, color=WHITE, spc=250))
    d.text(0.9, 1.9, 11.5, 1.1, P("Can LLMs reason?", size=58, family="serif", color=WHITE))
    d.text(0.9, 3.05, 11.5, 0.9, P("Testing and training language models with games on the Polaris "
                                   "supercomputer", size=26, family="serif", color=WHITE))
    d.line(0.9, 4.15, 6.5, 4.15, color=GREY_L, lw=1.0)
    d.text(0.9, 4.35, 11.5, 0.95, [
        P("**Decrypto** measures whether a model can give hints that its partner understands and its "
          "opponent does not.", size=16, color=GREY_L, sa=6),
        P("**MARSHAL** trains a model by having it play games against itself, then measures what "
          "changed.", size=16, color=GREY_L)])
    d.text(0.9, 5.75, 11.5, 0.35, P("[Presenter name]  ·  [Department], The University of Chicago",
                                    size=16, color=WHITE))
    d.text(0.9, 6.2, 11.5, 0.6, P("SC26  ·  Computing time on Polaris, Argonne Leadership Computing "
                                  "Facility (project lighthouse-uchicago)", size=12, color=GREY_L))


def slide2(d):
    d.slide(notes=NOTES[2])
    d.frame("QUESTION AND DESIGN",
            "We used games to test and to train reasoning, because a game scores itself")
    d.text(0.6, 1.92, 12.13, 0.85, P(
        "Exam-style benchmarks can leak into a model's training data, and grading free-form answers needs "
        "a person or a second model. We were geared towards problems that are new on every deal and that "
        "the rules alone can score, so we chose games. One project measured models as they are; the other "
        "trained a model and measured it again.", size=15, color=INK2, ls=1.05), name="intro")
    cards = [
        (0.6, DEC, "DECRYPTO  ·  MEASURE", [
            "**Question.** Can a model give hints that its partner understands and its opponent does not?",
            "**Design.** Four models. Each model's team played every model as opponent: 3 seeds × 32 "
            "games per pairing, 1,536 games, no training.",
            "**Machine.** Three 4-node jobs, one per seed. Each served three models and called the "
            "fourth over the internet."]),
        (6.83, MAR, "MARSHAL  ·  TRAIN", [
            "**Question.** Does a 4-billion-parameter model play better, including at games it never "
            "trained on, after playing against itself?",
            "**Design.** Two training runs of 200 steps (one game; three games), each checked every 10 "
            "steps on 15 fixed test conditions.",
            "**Machine.** 3 and 4 nodes; the longer run needed 34 starts on a shared queue."]),
    ]
    for x, col, label, paras in cards:
        d.rect(x, 2.95, 5.9, 2.75, line=GREY_L, lw=1.0)
        d.rect(x, 2.95, 5.9, 0.08, fill=col)
        d.text(x + 0.25, 3.18, 5.4, 0.25, P(label, size=11, bold=True, color=INK2, spc=150))
        d.text(x + 0.25, 3.55, 5.4, 2.05, [P(t, size=14, color=INK, sa=7, ls=1.0) for t in paras],
               name=label)
    d.text(0.6, 5.84, 12.13, 1.03, P(
        "**Polaris** (Argonne Leadership Computing Facility): each node has 4 NVIDIA A100 graphics "
        "processors (GPUs) with 40 GB of memory each. The supercomputing problem was fitting models and "
        "software written for 8-GPU servers with 80 GB cards, hundreds of simultaneous games, and "
        "multi-day runs onto a shared machine.", size=13.5, color=INK), fill=TINT, pad=0.15,
        name="polaris strip")


def slide3(d):
    d.slide(notes=NOTES[3])
    d.frame("DECRYPTO  ·  MEASURING REASONING",
            "The opponent cracked the code in 95% of 1,536 games, and games ended after about 4 rounds",
            marker=DEC)
    # game diagram
    d.rect(0.6, 1.98, 5.7, 1.3, line=DEC, lw=1.5, radius=0.08)
    d.text(0.8, 2.08, 5.3, 1.15, [
        P("**Encoder** (team)", size=14, sa=5),
        P("Sees four keywords:  1 boat · 2 direction · 3 viking · 4 foot", size=13, color=INK2, sa=3),
        P("Draws the code 3-1-4 and writes:  **norse · sail · step**", size=13, color=INK2)],
        name="encoder")
    d.line(2.0, 3.28, 2.0, 3.76, color=GREY_D, lw=1.25, arrow=True)
    d.line(4.9, 3.28, 4.9, 3.76, color=GREY_D, lw=1.25, arrow=True)
    d.rect(0.6, 3.78, 2.7, 1.38, line=DEC, lw=1.5, radius=0.08)
    d.text(0.78, 3.88, 2.38, 1.22, [
        P("**Decoder** (teammate)", size=14, sa=5),
        P("Knows the keywords. Guesses 3-1-4.", size=13, color=INK2)], name="decoder")
    d.rect(3.6, 3.78, 2.7, 1.38, line=GREY_D, lw=1.5, radius=0.08)
    d.text(3.78, 3.88, 2.38, 1.22, [
        P("**Interceptor** (opponent)", size=14, sa=5),
        P("Cannot see the keywords but keeps every earlier hint. Guesses the code too.", size=13,
          color=INK2)], name="interceptor")
    d.text(0.6, 5.36, 5.7, 0.55, P(
        "The team loses after 2 interceptions or 2 misreads, and wins by lasting 8 rounds.", size=13.5,
        color=INK), name="rules")
    d.text(0.6, 6.02, 5.7, 0.42, P(
        "Keywords and hints are from one of our logged games; the code and hint order are illustrative.",
        size=10.5, color=MUTED), name="caption")
    # chart: how the games ended
    cx, scale = 6.85, 4.3 / 1536
    d.text(cx, 1.98, 5.88, 0.3, P("How the 1,536 games ended", size=13, bold=True))
    rows = [("Opponent cracked the code twice", 1454, "1,454  (94.7%)", DEC),
            ("Team lasted all 8 rounds", 44, "44  (2.9%)", GREY_D),
            ("Team misread its own hints twice", 38, "38  (2.5%)", GREY_D)]
    y = 2.38
    for label, v, vl, col in rows:
        d.text(cx, y, 5.88, 0.24, P(label, size=12, color=INK2))
        bw = v * scale
        d.rect(cx, y + 0.28, bw, 0.22, fill=col)
        d.text(cx + bw + 0.08, y + 0.255, 1.6, 0.27, P(vl, size=12, bold=True, color=INK))
        y += 0.64
    d.text(cx, 4.35, 5.88, 1.38, [
        P("Games lasted 4.31 ± 0.03 rounds, against 4.33 ± 0.03 in the original study's model games "
          "(p = 0.65; UNVERIFIED) and 6.90 rounds for its 10 human games (UNVERIFIED).", size=13,
          color=INK, sa=6),
        P("The opponent set the outcome: it explained 49% of the variation in interceptions and 88% "
          "in game length. The team explained 53% of the variation in misreads.", size=13, color=INK)],
        name="results")
    d.therefore(cx, 5.82, 5.88, 0.72, "Therefore these models read hints well as partners and as "
                "eavesdroppers, but rarely wrote hints that kept an informed opponent out.", size=13.5)
    d.text(0.6, 6.6, 12.13, 0.36, P(
        "Models: Qwen2.5-72B, Qwen3-8B, Qwen3-4B, Claude Sonnet 4.6. A team is one model as both encoder "
        "and decoder. Settings from Lupu, Willi & Foerster (2025), temperature 0.6.", size=10.5,
        color=MUTED), name="footnote")


def slide4(d):
    d.slide(notes=NOTES[4])
    d.frame("DECRYPTO  ·  ON POLARIS",
            "We served four models from one 4-node job, and the slowest model set how many games "
            "could run at once", marker=DEC)
    # node diagram
    d.rect(0.6, 1.98, 2.45, 0.62, line=GREY_D, lw=1.0, radius=0.08)
    d.text(0.72, 2.03, 2.21, 0.52, [P("**Claude Sonnet 4.6**", size=11.5),
                                    P("Anthropic, over the internet", size=10.5, color=INK2)],
           anchor="m", name="claude")
    d.line(1.82, 3.0, 1.82, 2.62, color=GREY_D, lw=1.25, arrow=True)
    d.rect(0.6, 3.0, 2.45, 2.4, line=GREY_D, lw=1.0, radius=0.08)
    d.text(0.75, 3.1, 2.15, 0.25, P("**Node 0 · game runner**", size=11.5))
    d.gpu_row(0.75, 3.45, [None] * 4)
    d.text(0.75, 3.9, 2.15, 1.45, P("Runs 16 games at a time; no GPU work. Reaches Claude through the "
                                    "lab's web proxy.", size=11, color=INK2), name="node0")
    nodes = [(1.98, "Node 1 · Qwen2.5-72B", [DEC] * 4, "136 GB of weights: 34 of 40 GB on each GPU"),
             (3.48, "Node 2 · Qwen3-8B", [DEC, None, None, None], "One GPU"),
             (4.73, "Node 3 · Qwen3-4B", [DEC, None, None, None], "One GPU")]
    for y, label, fills, desc in nodes:
        hh = 1.32 if fills.count(DEC) == 4 else 1.05
        d.rect(3.55, y, 2.85, hh, line=GREY_D, lw=1.0, radius=0.08)
        d.text(3.7, y + 0.08, 2.6, 0.25, P(f"**{label}**", size=11.5))
        d.gpu_row(3.7, y + 0.4, fills)
        d.text(3.7, y + 0.78, 2.6, hh - 0.82, P(desc, size=10.5, color=INK2), name=label)
        d.line(3.05, 4.2, 3.55, y + hh / 2, color=GREY_D, lw=1.0, arrow=True)
    d.text(0.6, 5.88, 5.8, 0.38, P("Filled squares are GPUs holding a model: 6 of 16. Polaris "
                                   "allocates whole nodes.", size=10.5, color=MUTED), name="caption")
    # findings
    rx, rw = 6.9, 5.83
    d.text(rx, 1.98, rw, 1.0, P(
        "**Memory.** The 72-billion-parameter model fit only when split across all 4 GPUs of a node. "
        "What remained held 18,800 tokens of conversation, about 2.3 full-length requests at once, and "
        "the model loaded on our third configuration.", size=12.5), name="memory")
    d.text(rx, 3.05, rw, 0.26, P("Games completed in the earlier 3-model run (405 games)", size=12,
                                 bold=True))
    scale = 3.3 / 405
    y = 3.38
    for label, v, vl, col in [("All 405 games at once", 245, "245", GREY_D),
                              ("24 games at a time", 403, "403  (+2 after a parser fix)", DEC)]:
        d.text(rx, y, rw, 0.22, P(label, size=11, color=INK2))
        d.rect(rx, y + 0.25, v * scale, 0.2, fill=col)
        d.text(rx + v * scale + 0.08, y + 0.225, 2.4, 0.25, P(vl, size=11, bold=True))
        y += 0.56
    d.text(rx, 4.55, rw, 0.78, P(
        "**Concurrency.** With all 405 games at once, the 72B server dropped 610 of 1,929 requests (32%) "
        "and 160 games were lost. With 24 at a time it dropped 0 of 1,736, and the run took 2 h 27 min.",
        size=12.5), name="concurrency")
    d.text(rx, 5.36, rw, 0.92, P(
        "**Scheduling.** Separate server and game jobs started hours apart, and the servers ran out their "
        "time limits idle. One job holding all four nodes starts them together. A 10-hour request started "
        "2.5 h after submission; a 16-hour request had waited 32 h with no start time.", size=12.5),
        name="scheduling")
    d.therefore(0.6, 6.32, 12.13, 0.58, "Therefore the number of simultaneous games has to be set by the "
                "memory of the slowest model, not by the number of games to play.", size=13.5)


def slide5(d):
    d.slide(notes=NOTES[5])
    d.frame("MARSHAL  ·  TRAINING REASONING",
            "Self-play taught a 4-billion-parameter model to make legal moves within 10 steps; a better "
            "strategy showed up in only one game", marker=MAR)
    boxes = [(0.6, 1.98, "1 · Play", "The model plays both seats and reasons in text before each move."),
             (3.35, 1.98, "2 · Score", "The engine returns win, draw, loss, or Hanabi's team "
                                       "score. Illegal or over-long moves forfeit."),
             (3.35, 3.6, "3 · Learn", "Shift the model toward the moves that preceded wins."),
             (0.6, 3.6, "4 · Copy", "Send the updated model to the players, then play again.")]
    for x, y, t, body in boxes:
        d.rect(x, y, 2.35, 1.27, line=MAR, lw=1.5, radius=0.08)
        d.text(x + 0.15, y + 0.1, 2.05, 1.1, [P(f"**{t}**", size=13, sa=3),
                                              P(body, size=11.5, color=INK2)], name=t)
    d.line(2.95, 2.6, 3.35, 2.6, color=GREY_D, lw=1.25, arrow=True)
    d.line(4.52, 3.25, 4.52, 3.6, color=GREY_D, lw=1.25, arrow=True)
    d.line(3.35, 4.23, 2.95, 4.23, color=GREY_D, lw=1.25, arrow=True)
    d.line(1.77, 3.6, 1.77, 3.25, color=GREY_D, lw=1.25, arrow=True)
    d.text(0.6, 5.05, 5.1, 1.15, [
        P("**Trained on** TicTacToe, Kuhn poker and Mini Hanabi (a cooperative card game).", size=12,
          sa=4),
        P("**Tested** every 10 steps on games it never trained on (Connect Four, Leduc poker, Simple and "
          "Full Hanabi) against fixed opponents: a search player and poker-solver strategies.", size=12)],
        name="games")
    # dumbbell chart
    cx0, cx1 = 6.35, 12.35
    xs = lambda v: cx0 + (cx1 - cx0) * v / 70.0
    d.text(6.1, 1.98, 6.63, 0.28, P("Before and after training, three-game run (Qwen3-4B, 200 steps)",
                                    size=12.5, bold=True))
    d.rect(6.12, 2.36, 0.13, 0.13, fill=GREY_M, oval=True)
    d.text(6.3, 2.3, 1.2, 0.24, P("before", size=11, color=INK2))
    d.rect(7.12, 2.36, 0.13, 0.13, fill=MAR, oval=True)
    d.text(7.3, 2.3, 1.2, 0.24, P("after", size=11, color=INK2))
    top, rowh = 2.62, 0.74
    for t in (0, 20, 40, 60):
        d.line(xs(t), top, xs(t), top + 3 * rowh, color="E6E6E0", lw=0.75)
        d.text(xs(t) - 0.3, top + 3 * rowh + 0.03, 0.6, 0.22, P(f"{t}%", size=10, color=MUTED, align="c"))
    rows = [("Illegal-move forfeits, board games, model moving second (step 0 → 10–50)", 48.8, 0.8),
            ("TicTacToe draws in games played out, moving second, vs a 1,000-simulation search player "
             "(step 0 → 130–170)", 37.8, 59.6),
            ("Forfeits on the 8 never-trained test conditions (steps 130–170 → 180–190)", 24.7, 63.8)]
    for i, (label, a, b) in enumerate(rows):
        y = top + i * rowh
        d.text(6.1, y + 0.02, 6.63, 0.4, P(label, size=11, color=INK2), name=label)
        ly = y + 0.5
        d.line(xs(a), ly, xs(b), ly, color=GREY_M, lw=2.0)
        d.rect(xs(a) - 0.075, ly - 0.075, 0.15, 0.15, fill=GREY_M, oval=True)
        d.rect(xs(b) - 0.075, ly - 0.075, 0.15, 0.15, fill=MAR, oval=True)
        for v, bold in ((a, False), (b, True)):
            other = b if v == a else a
            if v >= other:
                d.text(xs(v) + 0.12, ly - 0.12, 0.8, 0.24, P(f"{v:.1f}%", size=10.5, bold=bold))
            else:
                d.text(xs(v) - 0.92, ly - 0.12, 0.8, 0.24, P(f"{v:.1f}%", size=10.5, bold=bold, align="r"))
    d.text(6.1, 5.12, 6.63, 1.12, [
        P("Mini Hanabi's team score rose from 1.05 (step 0) to 2.07 ± 0.06 (steps 130–170); the paper "
          "reports 2.17 (App. J; UNVERIFIED). The draw gain passed a false-discovery correction "
          "(q = 0.048); the first seat showed no gain.", size=11.5, sa=4),
        P("Leduc poker, never trained, did not improve: at steps 130–170 the three-game run won 23.5% of "
          "second-seat games, 20.9 ± 6.8 points below the one-game run (q = 0.038).", size=11.5)],
        name="results")
    d.therefore(0.6, 6.3, 12.13, 0.6, "Therefore self-play mostly taught the model to stop forfeiting "
                "(legal moves, shorter reasoning). The strategic gain is one game and one seat, and it did "
                "not reach Leduc poker.", size=13)


def slide6(d):
    d.slide(notes=NOTES[6])
    d.frame("MARSHAL  ·  ON POLARIS",
            "Across 4 nodes, generating games took 65% of each training step; copying the model between "
            "nodes took 0.2%", marker=MAR)
    # layout diagram
    d.rect(0.6, 1.98, 5.4, 1.5, line=MAR, lw=1.5, radius=0.08)
    d.text(0.75, 2.06, 5.1, 0.25, P("**Nodes 0–1 · learner**", size=12))
    d.gpu_row(0.75, 2.4, [MAR] * 4)
    d.gpu_row(2.35, 2.4, [MAR] * 4)
    d.text(3.95, 2.36, 1.95, 1.05, P("Model split 4 ways in each node, 2 copies across nodes; a frozen "
                                     "reference copy shares these GPUs.", size=10.5, color=INK2),
           name="learner")
    d.text(0.75, 2.8, 1.4, 0.22, P("node 0", size=9.5, color=MUTED))
    d.text(2.35, 2.8, 1.4, 0.22, P("node 1", size=9.5, color=MUTED))
    d.line(1.9, 3.5, 1.9, 4.08, color=GREY_D, lw=1.25, arrow=True)
    d.line(4.2, 4.08, 4.2, 3.5, color=GREY_D, lw=1.25, arrow=True)
    d.text(2.0, 3.66, 2.0, 0.3, P("updated model: 2.1 s", size=10.5, color=INK2))
    d.text(4.3, 3.66, 1.7, 0.3, P("scored games", size=10.5, color=INK2))
    d.rect(0.6, 4.1, 5.4, 1.3, line=GOLD, lw=1.5, radius=0.08)
    d.text(0.75, 4.18, 5.1, 0.25, P("**Nodes 2–3 · game players**", size=12))
    d.gpu_row(0.75, 4.52, [GOLD] * 4)
    d.gpu_row(2.35, 4.52, [GOLD] * 4)
    d.text(3.95, 4.48, 1.95, 0.85, P("8 game-playing engines, 1 GPU each; 192 games per step (64 of "
                                     "each game).", size=10.5, color=INK2), name="engines")
    d.text(0.6, 5.55, 5.4, 0.68, P(
        "The released code assumes one 8-GPU server (the paper used 80 GB H100s) with every role on every "
        "GPU. On 40 GB cards we split the roles across nodes.", size=10.5, color=MUTED), name="note")
    # stacked bar: one step
    bx, bw, tot = 6.4, 6.33, 912.0
    d.text(bx, 1.98, bw, 0.28, P("One training step: 15.2 min (mean of 176 steps)", size=12.5, bold=True))
    segs = [("Generating games", 593, GOLD, INK), ("Learning", 219, MAR, WHITE),
            ("Re-scoring", 88, GREY_D, WHITE), ("Copy", 2.1, GREY_M, None), ("rest", 9.9, GREY_L, None)]
    x = bx
    by, bh = 2.36, 0.42
    for name, v, col, tc in segs:
        sw = bw * v / tot
        d.rect(x, by, max(sw - 0.02, 0.012), bh, fill=col)
        if name == "Generating games":
            d.text(x + 0.1, by, sw - 0.2, bh, P("Generating games  593 s", size=11, bold=True, color=tc),
                   anchor="m")
        elif name == "Learning":
            d.text(x + 0.06, by, sw - 0.1, bh, P("Learning  219 s", size=10, bold=True, color=tc,
                                                 align="c"), anchor="m")
        elif name == "Re-scoring":
            d.text(x + 0.03, by, sw - 0.05, bh, P("88 s", size=10.5, bold=True, color=tc, align="c"),
                   anchor="m")
        x += sw
    d.text(bx, 2.86, bw, 0.62, [
        P("Learning: updating the model from the scored games. Re-scoring: the current and frozen copies "
          "re-read each game (2 × 44 s). Copying the updated model to the players: 2.1 s.", size=10.5,
          color=INK2)], name="legend")
    d.text(bx, 3.58, bw, 0.92, P(
        "**Network.** A fallback from Polaris's Slingshot network to ordinary TCP looks normal in the logs, "
        "so each job fails unless the Slingshot provider is selected. Measured across 2 nodes: 36 GB/s "
        "all-reduce bandwidth; at the measured 21 GB/s point-to-point rate, an 8 GB model copy takes "
        "about 0.4 s.", size=12), name="network")
    d.text(bx, 4.53, bw, 0.85, P(
        "**Memory.** At 32,768 tokens per sequence the learner ran out of memory until it took one "
        "sequence at a time. About 80% of training tokens were padding; trimming it cut learning from 253 s to "
        "68 s per step in the 3-node run.", size=12), name="memory")
    d.text(bx, 5.41, bw, 0.85, P(
        "**Idle time.** Play and learning take turns, so in the final attempt (steps 105–199) GPUs were "
        "busy 29% of the time on the learner nodes and 54% on the game nodes. Each step also waited for "
        "its slowest game: 221 of 230 collections waited 5 min or more, TicTacToe the last in 207.",
        size=12), name="idle")
    d.therefore(0.6, 6.32, 12.13, 0.58, "Therefore the cost of this training is generating games, not "
                "communication. We hypothesize that overlapping play with learning, not a faster network, "
                "is the next speed-up.", size=13)


def slide7(d):
    d.slide(notes=NOTES[7])
    d.frame("BOTH PROJECTS  ·  A SHARED MACHINE",
            "Our 58 hours of compute took 10.6 days from submission; restarts kept 85% of the node time "
            "we held")
    d.text(0.6, 1.95, 12.13, 0.28, P(
        "MARSHAL three-game run from submission to finish, in hours (4 nodes, preemptable queue, "
        "13–23 October 2026)", size=12.5, bold=True))
    bx, bw, tot = 0.6, 12.13, 255.4
    by, bh = 2.62, 0.55
    segs = [(157.7, GREY_L, ["Waiting for the first start  ·  157.7 h (6.6 days)"], INK),
            (30.2, GREY_M, ["Between restarts", "30.2 h"], INK),
            (9.8, GREY_D, None, None),
            (57.7, MAR, ["Compute kept  ·  57.7 h"], WHITE)]
    x = bx
    for v, col, label, tc in segs:
        sw = bw * v / tot
        d.rect(x, by, sw - 0.025, bh, fill=col)
        if label:
            size = 11.5 if sw > 2 else 10
            d.text(x + 0.1, by, sw - 0.18, bh, [P(t, size=size, bold=True, color=tc) for t in label],
                   anchor="m", name=label[0])
        else:
            mid = x + sw / 2
            d.line(mid, by - 0.04, mid, by - 0.16, color=GREY_D, lw=1.0)
            d.text(mid - 3.2, by - 0.42, 3.4, 0.24, P("Running, work not kept  ·  9.8 h", size=10.5,
                                                      bold=True, color=INK, align="r"))
        x += sw
    for t in (0, 50, 100, 150, 200, 250):
        tx = bx + bw * t / tot
        d.line(tx, by + bh + 0.02, tx, by + bh + 0.08, color=GREY_D, lw=0.75)
        d.text(tx - 0.4, by + bh + 0.09, 0.8, 0.22, P(f"{t} h", size=10, color=MUTED, align="c"))
    # two columns
    d.text(0.6, 3.62, 5.9, 2.6, [
        P("What happened", size=14, bold=True, color=MAROON, sa=5),
        P("The run was stopped 33 times (34 starts). On one night 16 starts in a row were stopped after "
          "1 to 22 minutes, most before finishing a step. No hardware, memory or network errors were "
          "logged; urgent jobs reclaiming the nodes fits the pattern (UNVERIFIED).", size=13.5, sa=8),
        P("Decrypto's game runner had no resume. One seed's job was stopped 8 times and replayed from the "
          "start each time; it finished on its 9th start, in about 7 hours.", size=13.5)],
        name="what happened")
    d.text(6.83, 3.62, 5.9, 2.6, [
        P("What made it finish", size=14, bold=True, color=MAROON, sa=5),
        P("The scheduler requeued the job after each stop (it was submitted as rerunnable). It saved a "
          "complete checkpoint every 5 steps (40 saves) and on restart loaded the newest complete one, "
          "including each GPU's random-number and optimizer state. The final history matched the attempts "
          "that produced it in all 64,537 logged values.", size=13.5,
          sa=8),
        P("Cost: 270 node-hours (1,080 GPU-hours) for 200 steps. The one-game run (3 nodes) used about "
          "120 node-hours in a single 40.2\u00a0h allocation.", size=13.5)], name="what made it finish")
    d.therefore(0.6, 6.32, 12.13, 0.58, "Therefore time to result was set by the queue and by restarts, "
                "not by compute speed; complete, frequent checkpoints decided how much held time was kept.",
                size=13)


def slide8(d):
    d.slide(notes=NOTES[8])
    d.frame("INTERPRETATION, LIMITS AND VERDICT",
            "Models read hints and learned rules quickly; we found little evidence that they model an "
            "opponent")
    d.text(0.6, 1.92, 4.3, 3.95, [
        P("Interpretation", size=13, bold=True, color=MAROON, sa=5),
        P("**Decrypto.** Misreads were rare (2.5%) and set by the team; interceptions were common (94.7%) "
          "and set by the opponent. Therefore the gap is in writing hints an informed eavesdropper cannot "
          "decode, not in being understood by a partner.", size=13, sa=9),
        P("**MARSHAL.** Forfeits fell first and fastest; the one strategic gain was TicTacToe as second "
          "mover. Therefore self-play mostly taught the model to stop forfeiting. We hypothesize the late "
          "breakdown comes from a fragile model or from game engines that ran 26.8 h through 95 in-place "
          "updates; serving checkpoints 169, 189 and 199 in fresh engines would test it.", size=13)], name="interpretation")
    lessons = [
        "Set concurrency from the slowest model's memory: 160 lost games became 0.",
        "Put every part of a pipeline in one job, so the parts start and stop together.",
        "Fail the job when traffic leaves the fast network; the fallback is silent.",
        "Checkpoint completely and often, and requeue automatically: 85% of held node time was kept "
        "across 34 starts.",
        "Measure where each step's time goes: generating games took 65%, copying the model 0.2%."]
    d.text(5.15, 1.92, 4.0, 3.95,
           [P("What carries to other HPC workloads", size=13, bold=True, color=MAROON, sa=5)] +
           [P(f"{i}.\t{t}", size=13, sa=7, hang=(0.3, 0.3)) for i, t in enumerate(lessons, 1)],
           name="lessons")
    limits = [
        "One base model and one training seed per run; no ablations.",
        "Paper and human comparison numbers are UNVERIFIED (transcribed, not recomputed).",
        "Claude ran through its command-line tool at default sampling with a 180 s timeout, not the "
        "study's settings.",
        "The three-game run's late windows were chosen after seeing the data."]
    d.text(9.4, 1.92, 3.33, 3.95,
           [P("Limits", size=13, bold=True, color=MAROON, sa=5)] +
           [P(f"•\t{t}", size=13, sa=7, hang=(0.2, 0.2)) for t in limits], name="limits")
    d.text(0.6, 5.88, 12.13, 1.0, [
        P("**Verdict: partly.** Current models reason well enough to decode hints and to learn rules from "
          "play; in our runs they did not reliably anticipate what an opponent can infer. Next, we place "
          "the self-play-trained model (checkpoint 149) in Decrypto against the same opponents and seeds "
          "as the untrained model.",
          size=13.5, color=WHITE)], fill=MAROON, pad=0.14, anchor="m", name="verdict")


# --------------------------------------------------------------------------- notes
NOTES = {
1: """
Speaker notes. This talk covers two projects that ran on Polaris at the Argonne Leadership Computing Facility under project lighthouse-uchicago. Decrypto measured how current language models play a word game that needs reasoning about what other players know. MARSHAL trained a 4-billion-parameter model by self-play and measured what changed. The supercomputing thread is how each workload was fitted onto 40 GB A100 nodes and a shared queue.
Before presenting: replace the presenter and department placeholders, and confirm the conference label (SC26).
Every number in the deck is sourced in the notes of the slide where it appears. Numbers taken from the original papers or their human games are tagged UNVERIFIED because we transcribed them rather than recomputed them.
""",
2: """
Question and design. We asked whether large language models reason, and whether training by play changes that. We were geared towards problems that are new on every deal and that the rules alone can score, so we chose games: no human grader and no second model acting as judge.
Decrypto: Lupu, Willi & Foerster, The Decrypto Benchmark for Multi-Agent Reasoning and Theory of Mind, arXiv:2506.20664 (2025). We used the paper's released competitive configuration (config/paper/figure_3_competitive.yaml) and changed only the roster: Qwen2.5-72B-Instruct, Qwen3-8B, Qwen3-4B and Claude Sonnet 4.6. 4 teams x 4 opponents x 3 seeds x 32 episodes = 1,536 games (jobs 7668935-7668937).
MARSHAL: Incentivizing Multi-Agent Reasoning via Self-Play with Strategic LLMs, arXiv:2510.15414. R5 trained on TicTacToe alone on 3 nodes (job 7668069); R6 trained on TicTacToe, Kuhn poker and Mini Hanabi on 4 nodes (job 7741511). Both validated every 10 steps on the same 15 conditions, 1,500 trajectories per pass, seed 42.
Hardware: 4 x A100-SXM4-40GB per node, confirmed with nvidia-smi in job 7197265 (decrypto/polaris_pbs_notes.md). The MARSHAL upstream configuration assumes one server with 8 x 80 GB GPUs and every role colocated (MARSHAL/polaris_multinode_slingshot.md section 0). Up to 405 Decrypto games ran at once in job 7199012, and a MARSHAL validation pass runs 480 environments.
""",
3: """
Results. As the chart shows, interceptions ended 1,454 of 1,536 games, an interception share of 0.947 +/- 0.006 (mean over match-ups and seeds +/- SE). As in the paper's analysis, this count includes games where both limits were reached on the same turn. The team survived 44 games (2.9%) and lost 38 (2.5%) to miscommunication. Games lasted 4.31 +/- 0.03 turns. The paper's LLM-only block lasted 4.33 +/- 0.03 turns (UNVERIFIED; z = -0.45, p = 0.65). Its interception share was 0.904 +/- 0.007 (UNVERIFIED), lower than ours by 0.043 (p = 3e-6). With each study's weakest opponent removed, the difference fell to +0.008 (p = 0.19).
In a two-way analysis of variance over the 48 match-up x seed values, the opponent accounted for eta-squared = 0.49 of the variation in interception share (F(3,32) = 18.0, p = 5e-7) and 0.88 of the variation in game length (F(3,32) = 240). The team accounted for 0.53 of the variation in miscommunication share (F(3,32) = 18.8, p = 3e-7). Claude's team never miscommunicated (0 of 384 games) and survived 6.2% of its games; Qwen3-4B's team miscommunicated in 6.8%.
Human teams in the paper lasted 6.90 turns and won 4 of 10 games against Llama-3.1-70B (UNVERIFIED).
Therefore the interceptor's task, matching three new hints to four growing lists of earlier hints, was easy for these models, and teams rarely wrote hints that defeated it.
Limits: Claude ran through the claude -p command-line tool at its default sampling, with a flattened transcript and a 180 s timeout (73 calls timed out and went through the game's retry path). The example keywords and hints are from the 3-model run log (job 7199012); the code and hint order are illustrative.
Source: decrypto/polaris_4model_competitive_results.md (analysis job 7731552); data in results/polaris_4model_competitive_s{0,1,2}/experiment_summary.csv.
""",
4: """
Design. Node 0 ran the game loop and called Claude through the ALCF web proxy; nodes 1-3 each served one model with vLLM. One PBS job held all four nodes (slurm/fused_4model_polaris.pbs) and started the servers with mpiexec.
Memory: Qwen2.5-72B weights are 136 GiB, which needs 4 GPUs (33.98 GiB per GPU, measured at load). The first two configurations failed with no memory left for the conversation cache (jobs 7197372, 7197374). The third loaded (job 7197375): memory-utilization 0.97, at most 64 sequences, prefill batch 2,048 tokens. That left a cache of 18,800 tokens, 2.29 requests at the 8,192-token context limit.
Concurrency, earlier 3-model run: job 7199012 ran all 405 games at once. The 72B server aborted 610 of 1,929 requests, the client logged 312 timeouts, and 160 games were dropped, so 245 of 405 finished. Job 7199082 capped the run at 24 games: 0 aborts of 1,736 requests, 403 of 405 games, 8,851 s of run time. The last 2 games failed on an empty model response that crashed the answer parser. We fixed the parser and reran them (job 7199219), reaching 405 of 405.
Scheduling: separate server jobs died at their walltime while the dependent game job was still queued (jobs 7197359, 7197380, 7197381; queue skew of 3.4-7 h). The fused job 7197574 started 1 minute after submission on the capacity queue and ran 27 games in 25 min. A 16 h request (7197605) waited about 32 h with no estimated start. The 10 h resubmission (7199012) started about 2.5 h after submission.
Network gotcha (4-model run): node 0 needed direct traffic to the servers and proxied traffic to Claude. The HTTP client ignored address ranges in no_proxy, and each node name resolves to two addresses, so requests went to the proxy (jobs 7599882, 7668624 failed). Listing each registered server address exactly, with a check that fails the job if any would still be proxied, fixed it (job 7668797).
Cost: the 405-game production run used about 10 node-hours.
Sources: decrypto/polaris_scaleup_gpu_needs.md; decrypto/polaris_pbs_notes.md (job ledger).
""",
5: """
Design. We ran the authors' released training code on Polaris with two paper-directed changes: 128 games' worth of trajectories from each game per step, and the TicTacToe reward doubled. Settings: Qwen3-4B, temperature 0.6, at most 4,096 tokens per turn, thinking forced, 200 steps. Windows: step 0 (before training), steps 10-50, steps 130-170 (late but healthy; chosen after seeing the data) and steps 180-190 (after the breakdown; also chosen after seeing the data).
Results. Malformed-move forfeits with the model moving second in TicTacToe and Connect Four fell from 48.8 +/- 2.4% at step 0 to 0.8 +/- 0.5% over steps 10-50. The one-game run (R5) went from 50.5% to 1.8%. Mini Hanabi's team score rose from 1.05 to 2.07 +/- 0.06 (steps 130-170), against 2.17 for the paper's generalist (UNVERIFIED). The Mini Hanabi gain came mainly from fewer over-long turns, not from better move formatting. In TicTacToe as second mover against a search player using 1,000 simulations per move (Monte Carlo tree search), draws among games played to the end rose from 37.8% to 59.6% (+21.8 +/- 7.2 points, p = 0.0027, q = 0.048; analysis/r6/out/conditional_tests.csv). The first seat showed no gain. Leduc poker, never trained, did not improve. At steps 130-170 the three-game run won 23.5 +/- 5.2% of second-seat games against 44.4 +/- 4.4% for the one-game run (difference -20.9 +/- 6.8 points, q = 0.038; r6_vs_r5_rates.csv). At steps 180-190 the gap was 32.6 points. The within-run drop from step 0 (40.2% to 23.5%, -16.7 points, p = 0.014) does not survive the false-discovery correction (q = 0.13; window_tests.csv). Its 40.2% baseline is also the one outlier in the step-0 replicate check: the identical untrained model scored 26.6% in the one-game run's step-0 pass. We therefore show the between-run contrast, not the within-run drop. The paper reports a 28.7% gain on Leduc for its generalist (UNVERIFIED); we did not reproduce it. The 2.17 Mini Hanabi score is the paper's App. J value as transcribed in marshal_r6_results.md Table 1.
Breakdown: from about step 172, with the learning rate at or below 4.4e-8, the run degraded in every game. Self-play TicTacToe forfeits rose from 3.6% to 43.1%, token entropy from 0.93 to 2.26 nats, the divergence from the base model from 0.18 to 0.69, and forfeits on the 8 held-out conditions (Connect Four x4, Leduc x2, Simple and Full Hanabi; family_rates.csv) from 24.7 +/- 8.8% to 63.8 +/- 15.5%. The 180-190 window was chosen after seeing the breakdown. The weights did not move faster: relative change per 5 steps fell from 3.5e-4 to 2.5e-5. Checkpoint 149 is the last clean model.
Source: MARSHAL/marshal_r6_results.md (R1, R2, R5, R6, R7, R8; analysis job outputs in analysis/r6/out/). The Marshall_multinode session fact-checked every MARSHAL line against those files on 2026-11-13, and its corrections are applied.
""",
6: """
Layout (MARSHAL/marshal_r6_generalist.md section 1; config agentic_val_multi_games_selfplay_polaris_megatron_prod.yaml): the learner and the frozen reference copy on GPUs 0-7 (nodes 0-1; tensor-parallel 4 within a node x data-parallel 2 across nodes). Eight vLLM engines on GPUs 8-15 (nodes 2-3), one GPU each. Each step collects 384 trajectories: 128 per game, and each self-play game yields one trajectory per seat, so 192 games (64 per game). ROLL launches one driver per node with mpiexec; rank 0 becomes the Ray head.
Step time (marshal_r6_results.md R9, Table 5): a non-validation step took 15.2 +/- 1.1 min (n = 176): game generation 593 +/- 48 s, learning 219 +/- 16 s, the two re-scoring passes 44 s each, and the model copy 2.1 s. Shares are computed against 912 s (15.2 min); about 10 s is not itemized. The 593 s of game generation includes waiting for the slowest game to reach its 128 trajectories.
Network (MARSHAL/polaris_multinode_slingshot.md section 0a, job 7666124, 2 nodes): all-reduce of 1 GiB at 36.35 GB/s bus bandwidth. The 36 GB/s is all-reduce bus bandwidth over 8 GPUs on 2 nodes. A two-rank cross-node 256 MB transfer ran at 20.64 GB/s, so an 8 GB model copy would take about 0.4 s (an estimate, not a timed copy). The whole model-copy step in R6 took 2.1 s. Recipe: aws-ofi-nccl v1.6.0 + libfabric 2.3.1 + NCCL_PROTO=Simple, plus a guard that fails the job unless NCCL logs "Selected Provider is cxi".
The released configuration assumes one 8-GPU node with every role on every GPU; the 80 GB figure comes from the paper's hardware (8 x H100).
Memory: job 7666482 ran out of memory at micro-batch 2 (output logits alone need 9.27 GiB per micro-batch). Micro-batch 1 with gradient accumulation 4 fixed it (job 7666705). ROLL padded every sequence to 32,768 tokens, about 80% padding; trimming it cut the learning phase from 253 s to 68 s and the step from 12.2 to 7.5 min (jobs 7666705 -> 7666858, 3-node layout).
Idle time: GPU-busy is nvidia-smi utilization.gpu, not streaming-multiprocessor occupancy: 29% on the learner nodes and 54% on the game nodes in R6's final attempt (steps 105-199), and 13% and 76% in R5. 221 of 230 game collections (230 includes reruns after restarts) waited 300 s or more for the per-game quota, and TicTacToe was the last game to fill in 207 of those 221 (analysis/r6/out/summary.json, "quota").
Process cap: Polaris allows 4,096 processes per node; the peak was 3,350 (R6). Lowering a thread pool to stay under the cap deadlocked at 256 or more in-flight requests (job 7667241), fixed by giving responses their own thread group (job 7667360).
We hypothesize that overlapping generation with learning would cut the step time; we have not tested it.
""",
7: """
Results (MARSHAL/marshal_r6_results.md R10, Table 6; out/attempts.csv). Job 7741511 was submitted 2026-10-13 05:08 UTC, first started 2026-10-19 18:49, and finished 2026-10-23 20:31 (Exit 0, 200 of 200 steps). That is 157.7 h in the queue and then 97.7 h to the end. The 34 attempts held 67.5 h of 4-node allocation (270 node-hours), of which 85.5% (57.7 h) went into steps that the final checkpoint lineage kept. The 57.7 h includes 7.0 h of validation passes, so the slide calls it compute rather than training. The bar's middle segments are our arithmetic: 30.2 h between restarts = 97.7 - 67.5; 9.8 h not kept = 67.5 - 57.7.
Attempts: 6 of 34 contributed lineage steps; 15 never finished a step; 28 of 33 killed attempts banked no new checkpoint. Attempts 17-32 ran on one 4-node set and were killed after 1-22 min; attempt 33 on the same set then ran 9.1 h cleanly. No killed attempt logged a CUDA, Xid, ECC, NCCL, out-of-memory or Ray error. All implicated nodes carry queue_tags demand,preemptable. Preemption by urgent demand-queue jobs fits, but PBS accounting is not readable from a user account (UNVERIFIED).
Restart: the job was submitted as rerunnable (qsub -r y), so PBS requeued it after each kill. It saved a checkpoint every 5 steps (save_steps 5; 40 saves). Multi-node auto-resume restores each rank's random-number state and optimizer shard (resume test job 7667188). The final run's logged history matched the attempts that produced each step: 64,537 values, maximum difference 0. Steps rerun after a resume were not bitwise reproducible (step 65 ran twice, with draw rates 0.39 and 0.44).
Decrypto: the seed-2 job (7668937) was preempted 8 times and finished on its 9th attempt, about 7 h. The runner has no resume, so each attempt replayed from scratch (decrypto/polaris_pbs_notes.md ledger).
R5 (job 7668069): 40.2 h in one allocation on 3 nodes, about 120 node-hours, no preemption. GPU-hours = node-hours x 4.
""",
8: """
Verdict. The two projects agree on one point. Models handle the parts of reasoning that the rules make explicit: decoding hints, and producing legal moves after a few steps of practice. Neither project showed a reliable gain in anticipating what another player will infer.
Next experiment: the MARSHAL checkpoints were converted to Hugging Face format on 2026-11-02 (models/marshal/r6_generalist_step149, job 7808365; models/marshal/r5_tictactoe_step199, job 7808364). We will play them in Decrypto against the same opponents and seeds as the base Qwen3-4B in the 4-model grid. Still to decide: whether to serve them with MARSHAL's training sampling (4,096 tokens per turn, top-p 0.99, top-k 100) or with Decrypto's (2,000-token reasoning budget, top-p 0.95, top-k 20).
Limits in full: R5 and R6 each used one seed and no ablation. They ran the authors' released advantage estimator, not the paper's Eq. 3. The fresh-engine test of checkpoints 169, 189 and 199 has not been run. Against the serving-state explanation, an earlier attempt ran 60 in-place weight updates in one engine process without breaking down. Decrypto's embedding baselines, cooperative setting and theory-of-mind experiments were not run. Every paper and human-game number is UNVERIFIED.
HPC lessons, with their sources: concurrency (decrypto jobs 7199012 -> 7199082); one fused job (7197574); network guard (MARSHAL job 7666124); checkpoint and requeue (MARSHAL job 7741511); step-time breakdown (marshal_r6_results.md R9).
""",
}


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "can_llms_reason_sc26.pptx"))
    ap.add_argument("--preview", default=None, help="directory for PNG previews + overflow check")
    a = ap.parse_args()
    d = Deck(preview=bool(a.preview))
    for f in (slide1, slide2, slide3, slide4, slide5, slide6, slide7, slide8):
        f(d)
    d.save(a.out)
    print("wrote", a.out)
    if a.preview:
        for p in d.render(a.preview):
            print("preview", p)
        print(f"{len(d.warnings)} layout warning(s)")
        for w in d.warnings:
            print("  WARN", w)


if __name__ == "__main__":
    main()
