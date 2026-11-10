"""R3-S2: the two report figures built from the project's evidence: the methodology diagram and the split composition.

Both are written as PNG and SVG with a two-line interpretation, like the EDA figures. Every number printed on them comes from `reporting.facts` or from
`reports/r1/split_report.json`; nothing is typed in. Colours are the Okabe-Ito set (checked with the visualization palette validator: all checks pass, with a
contrast warning for orange that is answered by direct count labels on every segment and the same numbers in the page table).
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

INK, MUTED, EDGE = "#222222", "#6b6b6b", "#8a8a8a"
LANE_FILL = {"data": "#E8F1F8", "splits": "#FBF1DC", "models": "#E6F3EE", "eval": "#F8E9E1"}
COLORS = {"genuine": "#0072B2", "known": "#E69F00", "unseen_validation": "#009E73", "unseen_final": "#D55E00"}
LABELS = {"genuine": "Genuine", "known": "Known TTS (MeloTTS, VITS)", "unseen_validation": "Unseen validation (XTTS v2)",
          "unseen_final": "Unseen final test (ChatterboxTTS)"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt": "echoproof-r3-s2", "svg.fonttype": "none", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": False, "figure.dpi": 100})
    return plt


def _save(fig, plt, out_base: Path, lines: tuple[str, str], foot_height: float) -> dict:
    width = int(fig.get_figwidth() * 10.5)
    text = "\n".join(textwrap.fill(line, width=width, subsequent_indent="   ") for line in lines)
    fig.text(0.01, 0.012, text, fontsize=8.5, ha="left", va="bottom", color=INK)
    paths = []
    for ext in ("png", "svg"):
        kwargs = {"dpi": 150} if ext == "png" else {}
        path = f"{out_base}.{ext}"
        fig.savefig(path, metadata={"Software": None} if ext == "png" else {"Date": None}, **kwargs)
        paths.append(path)
    plt.close(fig)
    return {"files": paths, "interpretation": list(lines)}


def methodology_diagram(facts: dict, out_base) -> dict:
    """Four lanes, three steps each. The text of every box is filled from the facts."""
    from matplotlib.patches import FancyBboxPatch
    plt = _plt()
    t = lambda key: facts[key]["text"]                                                                     # noqa: E731
    lanes = [
        ("DATA", "data", [
            f"PhonemeDF 1.0.1\n{t('data.stems_total')} stems × {t('data.components')} sources\n({t('data.audio_files')} files, {t('data.audio_gb')} GB)",
            f"Admission\nspeaker ID taken from the stem\n{t('data.stems_excluded')} stems excluded",
            f"Research cohort\n{t('cohort.stems')} stems × {t('data.components')} ={t('cohort.recordings')} recordings\n16 kHz mono FLAC, {t('cohort.hours')} h"]),
        ("SPLITS AND INTEGRITY", "splits", [
            f"Speaker-grouped split\n{t('split.speakers_train')} / {t('split.speakers_validation')} / {t('split.speakers_test')} speakers\n(train / validation / test, seed {t('split.seed')})",
            f"Chunk manifest\n{t('chunks.window_s')} s windows, {t('chunks.stride_s')} s stride\n{t('clean.chunks')} chunks",
            f"Leakage checks\n{t('leak.check_kinds')} check types on both protocols\n{t('leak.violations')} violations"]),
        ("MODELS AND PROTOCOLS", "models", [
            "Four approaches\nLogistic regression, BiGRU,\nWavLM + head, AST (partial fine-tune)",
            "Protocol B first\ntrain MeloTTS + VITS, validate on XTTS,\nChatterboxTTS only in the final test",
            "B is locked, then Protocol A\nall four TTS families known\n(B-before-A firewall)"]),
        ("EVALUATION", "eval", [
            f"Validation roles (speakers)\nselection {t('split.speakers_selection')}, calibration {t('split.speakers_calibration')},\noperating {t('split.speakers_operating')}",
            "Lock\nmodel, temperature, threshold\nhashes recorded",
            "Sealed test, scored once\nknown, unseen generator,\nand corruption (D) panels"]),
    ]
    fig, ax = plt.subplots(figsize=(13, 7.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")
    box_w, box_h, x0, gap = 24.5, 9.6, 17.5, 3.2
    row_y = [49.5, 35.0, 20.5, 6.0]
    centres = []
    for r, (lane, kind, boxes) in enumerate(lanes):
        y = row_y[r]
        ax.text(0.6, y + box_h / 2, lane, ha="left", va="center", fontsize=9, color=MUTED, fontweight="bold")
        row = []
        for c, text in enumerate(boxes):
            x = x0 + c * (box_w + gap)
            ax.add_patch(FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.25,rounding_size=1.0", fc=LANE_FILL[kind], ec=EDGE, lw=1.0, zorder=2))
            head, *rest = text.split("\n")
            ax.text(x + box_w / 2, y + box_h - 1.6, head, ha="center", va="top", fontsize=9.5, color=INK, fontweight="bold", zorder=3)
            ax.text(x + box_w / 2, y + box_h - 4.1, "\n".join(rest), ha="center", va="top", fontsize=8.6, color=INK, linespacing=1.35, zorder=3)
            row.append((x, y))
        centres.append(row)
        for c in range(len(boxes) - 1):                                                                       # arrows within a lane
            xa, xb = row[c][0] + box_w + 0.35, row[c + 1][0] - 0.35
            ax.annotate("", xy=(xb, y + box_h / 2), xytext=(xa, y + box_h / 2), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2), zorder=1)
    for r in range(len(lanes) - 1):                                                                           # elbow from a lane's last box to the next lane's first box
        (xa, ya), (xb, yb) = centres[r][-1], centres[r + 1][0]
        start_x, start_y = xa + box_w / 2, ya - 0.3
        end_x, end_y = xb + box_w / 2, yb + box_h + 0.3
        mid_y = (start_y + end_y) / 2
        ax.plot([start_x, start_x, end_x], [start_y, mid_y, mid_y], color=MUTED, lw=1.2, zorder=1)
        ax.annotate("", xy=(end_x, end_y), xytext=(end_x, mid_y), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2), zorder=1)
    lines = (f"Pipeline of the study. Counts are read from the manifests and reports, not typed in. The corruption (D) panel uses specification version "
             f"{t('corr.version')} ({t('corr.hash')}...): {t('corr.conditions')}.".replace("`", ""),
             "This is the design. Stages after the Protocol B lock (calibration, threshold, sealed test, Protocol A) are not yet run, and Protocol C (external transfer) is not part of this delivery.")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return _save(fig, plt, Path(out_base), lines, 0.08)


def split_composition(root, out_base) -> dict:
    """Recordings per partition and validation role, by class family, for Protocol B (left) and Protocol A (right)."""
    from matplotlib.ticker import FuncFormatter
    plt = _plt()
    report = json.loads((Path(root) / "reports/r1/split_report.json").read_text(encoding="utf-8"))
    rows = [("Train", "train", "-"), ("Validation: selection", "validation", "selection"), ("Validation: calibration", "validation", "calibration"),
            ("Validation: operating", "validation", "operating"), ("Test", "test", "-")]
    speakers = report["speaker_counts"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1.0, 1.0]})
    summary = {}
    for ax, proto, cats in ((axes[0], "B", ("genuine", "known", "unseen_validation", "unseen_final")), (axes[1], "A", ("genuine", "known"))):
        support = report["protocols"][proto]["support"]
        names = {"genuine": "genuine", "known": "known_family", "unseen_validation": "unseen_validation", "unseen_final": "unseen_final"}
        ypos = list(range(len(rows)))[::-1]

        def value_of(part, role, cat):
            return support.get(f"{part}|{role}|{names[cat]}|{'genuine' if cat == 'genuine' else 'tts'}", {}).get("recordings", 0)

        maxtotal = max(sum(value_of(part, role, c) for c in cats) for _, part, role in rows)
        xmax = maxtotal * 1.85
        for y, (label, part, role) in zip(ypos, rows):
            left, total, parts = 0, 0, []
            for cat in cats:
                value = value_of(part, role, cat)
                if value:
                    ax.barh(y, value, left=left, height=0.62, color=COLORS[cat], edgecolor="white", linewidth=1.5, zorder=2)
                    if value >= 0.075 * xmax:                                  # label inside only where the segment is wide enough; the text beside the bar has every number
                        ax.text(left + value / 2, y, f"{value:,}", ha="center", va="center", fontsize=8.6, zorder=3,
                                color=INK if cat in ("known", "unseen_validation") else "white", fontweight="bold")
                    parts.append(f"{value:,}")
                left += value
                total += value
            spk = speakers[f"{part}/{role}"]
            ax.text(total + 0.012 * xmax, y, f"{total:,} recordings, {spk} speakers\n= " + " + ".join(parts), ha="left", va="center",
                    fontsize=8.2, color=MUTED, linespacing=1.3)
        ax.set_yticks(ypos)
        ax.set_yticklabels([r[0] for r in rows], color=INK)
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_xlabel("Recordings", color=INK)
        ax.set_title(f"Protocol {proto}: {report['protocols'][proto]['recordings']:,} recordings", loc="left", color=INK, fontsize=11)
        ax.tick_params(colors=MUTED)
        ax.grid(axis="x", alpha=0.25, zorder=0)
        summary[proto] = {"recordings": report["protocols"][proto]["recordings"], "speakers": report["speakers"]}
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in ("genuine", "known", "unseen_validation", "unseen_final")]
    fig.legend(handles, [LABELS[c] for c in ("genuine", "known", "unseen_validation", "unseen_final")], loc="upper center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.995))
    lines = (f"Every speaker belongs to exactly one row group, so no speaker appears on both sides of any split. In Protocol B the unseen generators appear only "
             f"in validation (XTTS v2) and in the final test (ChatterboxTTS). Protocol A uses every recording of the cohort, with all four TTS families in each partition.",
             "Counts and speaker numbers are read from the R1 split report. The sum beside each bar follows the legend order: genuine, known TTS, then the unseen family.")
    fig.tight_layout(rect=[0, 0.1, 1, 0.93])
    out = _save(fig, plt, Path(out_base), lines, 0.1)
    out["summary"] = summary
    return out
