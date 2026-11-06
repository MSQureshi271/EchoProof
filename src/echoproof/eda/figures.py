"""R2-S2 step 2 (local, no audio shards needed): figure families for the Protocol B training partition.

Inputs are the R2-S2 measurement files (recording_stats_B_train_v1.parquet, mean_logmel_B_train_v1.npz, examples_B_train_v1.npz) and
files already in the repository (split report, admission/materialize/repair reports, header inspection, R2-S1 MFCC features).
Every figure is written as PNG and SVG with a two-line interpretation printed on it and listed in interpretations.json. The numbers
in the interpretation lines are computed from the data, never typed in.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from echoproof.data.cohort import SAMPLE_RATE, stem_from_path
from echoproof.eda.measure import EXAMPLE_COMPONENTS, MANIFEST_NAME, OUTPUT_NAMES, SETTINGS

COMPONENTS = EXAMPLE_COMPONENTS
LABELS = {"genuine": "Genuine (LibriSpeech)", "melo": "MeloTTS", "vits": "VITS"}
COLORS = {"genuine": "#0072B2", "melo": "#E69F00", "vits": "#009E73"}       # Okabe-Ito, distinguishable for colour-blind readers
FOOT = [0, 0.15, 1, 1]            # tight_layout rectangle that leaves room for the interpretation footer
FIGURE_NAMES = ("01_class_generator_counts", "02_duration_histograms", "03_paired_waveform_logmel", "04_level_and_silence",
                "05_spectral_balance", "06_mfcc_distributions", "07_format_audit", "08_speaker_group_sizes", "09_exclusion_table")


class EdaError(RuntimeError):
    pass


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt": "echoproof-r2-s2", "svg.fonttype": "none", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100})
    return plt


def auc(a, b) -> float:
    """P(a > b) + 0.5 P(a == b): Mann-Whitney probability computed by ranks. 0.5 means no separation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("both groups need values")
    from scipy.stats import rankdata
    ranks = rankdata(np.concatenate([a, b]))
    return float((ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def separation(frame, column) -> float:
    """How well one column alone separates genuine from synthetic, folded to 0.5..1 (direction ignored)."""
    value = auc(frame.loc[frame["label"] == 0, column], frame.loc[frame["label"] == 1, column])
    return max(value, 1.0 - value)


def verdict(value: float) -> str:
    return "little" if value < 0.6 else "moderate" if value < 0.75 else "strong"


def _median(frame, column, component):
    return float(frame.loc[frame["component"] == component, column].median())


def _save(fig, plt, out_dir: Path, name: str, lines: tuple[str, str]) -> dict:
    import textwrap
    width = int(fig.get_figwidth() * 10.5)
    text = "\n".join(textwrap.fill(line, width=width, subsequent_indent="   ") for line in lines)
    fig.text(0.01, 0.012, text, fontsize=8.5, ha="left", va="bottom", color="#222222")
    for ext in ("png", "svg"):
        kwargs = {"dpi": 150} if ext == "png" else {}
        fig.savefig(out_dir / f"{name}.{ext}", metadata={"Software": None} if ext == "png" else {"Date": None}, **kwargs)
    plt.close(fig)
    return {"figure": name, "files": [f"{name}.png", f"{name}.svg"], "interpretation": list(lines)}


# ---- inputs ------------------------------------------------------------------------------------------------------------------

def load_inputs(root, data_dir=None) -> dict:
    import pandas as pd
    root = Path(root)
    data_dir = Path(data_dir) if data_dir else root / "data" / "eda" / "r2_s2_B_train"
    manifest_path = data_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise EdaError(f"{manifest_path} not found: run the R2-S2 measurement step first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, spec in manifest["outputs"].items():
        digest = hashlib.sha256((data_dir / spec["file"]).read_bytes()).hexdigest()
        if digest != spec["sha256"]:
            raise EdaError(f"{spec['file']} does not match the hash in {MANIFEST_NAME}")
    frame = pd.read_parquet(data_dir / OUTPUT_NAMES["stats"])
    with np.load(data_dir / OUTPUT_NAMES["mel_mean"], allow_pickle=False) as z:
        mel_ids, mel_mean = z["recording_id"].tolist(), z["mel_mean"].copy()
    if mel_ids != frame["recording_id"].tolist():
        raise EdaError("mean log-mel rows are not aligned with the statistics table")
    if set(frame["component"]) != set(COMPONENTS):
        raise EdaError(f"unexpected components {sorted(set(frame['component']))}: this EDA is B train only")
    examples = dict(np.load(data_dir / OUTPUT_NAMES["examples"], allow_pickle=False))
    reports = root / "reports" / "r1"
    admission = json.loads((reports / "admission.json").read_text(encoding="utf-8"))
    return {"manifest": manifest, "frame": frame, "mel_mean": mel_mean, "examples": examples, "admission": admission,
            "split_report": json.loads((reports / "split_report.json").read_text(encoding="utf-8")),
            "materialize": json.loads((reports / "materialize.json").read_text(encoding="utf-8")),
            "repair": sorted(reports.glob("repair_*.json")),
            "headers_path": root / admission["inputs"]["inspection_dir"] / "headers.jsonl",
            "features_path": root / "data" / "features" / "r2_s1_B" / "lr_train_crops_B_v1.npz"}


# ---- figures -----------------------------------------------------------------------------------------------------------------

def fig_counts(inp, out_dir, plt):
    frame = inp["frame"]
    support = inp["split_report"]["protocols"]["B"]["support"]
    parts = {}
    for key, value in support.items():
        partition, _, _, label = key.split("|")
        parts.setdefault(partition, {"genuine": 0, "synthetic": 0})["genuine" if label == "genuine" else "synthetic"] += value["recordings"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1, 1.1]})
    counts = [int((frame["component"] == c).sum()) for c in COMPONENTS]
    bars = a.bar([LABELS[c] for c in COMPONENTS], counts, color=[COLORS[c] for c in COMPONENTS])
    a.bar_label(bars, fmt="{:,.0f}")
    a.set_ylabel("Recordings in B train")
    a.set_title("B train by generator")
    a.set_ylim(0, max(counts) * 1.15)
    order = [p for p in ("train", "validation", "test") if p in parts]
    gen, syn = [parts[p]["genuine"] for p in order], [parts[p]["synthetic"] for p in order]
    b.bar(order, gen, color=COLORS["genuine"], label="Genuine")
    b.bar(order, syn, bottom=gen, color="#CC79A7", label="Synthetic")
    for i, p in enumerate(order):
        b.text(i, gen[i] + syn[i], f"{gen[i] + syn[i]:,}", ha="center", va="bottom")
    b.set_ylabel("Recordings (class totals)")
    b.set_title("Protocol B recordings by partition (counts only)")
    b.legend(frameon=False)
    b.set_ylim(0, max(g + s for g, s in zip(gen, syn)) * 1.15)
    n_gen, n_syn = counts[0], counts[1] + counts[2]
    ratios = ", ".join(f"{p} {parts[p]['synthetic'] / parts[p]['genuine']:.1f}" for p in order)
    lines = (f"B train holds {sum(counts):,} recordings: {n_gen:,} genuine, {counts[1]:,} MeloTTS, {counts[2]:,} VITS, so synthetic is {n_syn / n_gen:.1f} times the genuine count.",
             f"Synthetic-to-genuine ratio by partition: {ratios}. No ChatterboxTTS recording is in train, and the right panel shows class totals only, not test content.")
    fig.tight_layout(rect=FOOT)
    return _save(fig, plt, out_dir, FIGURE_NAMES[0], lines), {"genuine": n_gen, "synthetic": n_syn}


def fig_duration(inp, out_dir, plt):
    frame = inp["frame"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    top = float(frame["duration_s"].quantile(0.995))
    bins = np.linspace(0, top, 41)
    for c in COMPONENTS:
        a.hist(frame.loc[frame["component"] == c, "duration_s"], bins=bins, histtype="step", linewidth=2, color=COLORS[c], label=LABELS[c])
    a.axvline(4.0, color="#444444", linestyle="--", linewidth=1)
    a.text(4.05, a.get_ylim()[1] * 0.92, "4 s crop", fontsize=8)
    beyond = int((frame["duration_s"] > top).sum())
    a.set_xlabel(f"Duration (s)  [{beyond} recordings longer than {top:.1f} s not shown]")
    a.set_ylabel("Recordings")
    a.set_title("Duration by generator")
    a.legend(frameon=False)
    b.boxplot([frame.loc[frame["component"] == c, "duration_s"] for c in COMPONENTS], tick_labels=[c.capitalize() for c in COMPONENTS], showfliers=False,
              patch_artist=True, boxprops={"alpha": 0.7})
    for patch, c in zip(b.patches, COMPONENTS):
        patch.set_facecolor(COLORS[c])
    b.set_ylabel("Duration (s)")
    b.set_title("Median and quartiles")
    short = {c: float((frame.loc[frame["component"] == c, "duration_s"] <= 4.0).mean()) for c in COMPONENTS}
    value = separation(frame, "duration_s")
    lines = (f"Median duration: genuine {_median(frame, 'duration_s', 'genuine'):.1f} s, MeloTTS {_median(frame, 'duration_s', 'melo'):.1f} s, VITS {_median(frame, 'duration_s', 'vits'):.1f} s; "
             f"recordings of 4 s or less: {short['genuine']:.1%}, {short['melo']:.1%}, {short['vits']:.1%}.",
             f"Duration alone separates genuine from synthetic with AUC {value:.2f} ({verdict(value)} separation); "
             + ("it is a shortcut candidate and R2-S3 must compare duration-matched subsets." if value >= 0.6 else "R2-S3 still tests it with duration-matched subsets."))
    fig.tight_layout(rect=FOOT)
    return _save(fig, plt, out_dir, FIGURE_NAMES[1], lines), {"duration_auc": value}


def fig_paired(inp, out_dir, plt):
    from echoproof.features.acoustic import log_mel
    import librosa
    ex = inp["examples"]
    centers = librosa.mel_frequencies(n_mels=66, fmin=0.0, fmax=SAMPLE_RATE / 2)[1:-1]
    results = []
    for stem in ex["stems"].tolist():
        waves = {c: ex[f"wave__{stem}__{c}"].astype(np.float32) / 32768.0 for c in COMPONENTS}
        mels = {c: log_mel(w) for c, w in waves.items()}
        span = max(len(w) for w in waves.values()) / SAMPLE_RATE
        amp = max(float(np.abs(w).max()) for w in waves.values()) * 1.05
        lo, hi = np.percentile(np.concatenate([m.ravel() for m in mels.values()]), [1, 99.5])
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), sharex=True)
        image = None
        for j, c in enumerate(COMPONENTS):
            t = np.arange(len(waves[c])) / SAMPLE_RATE
            axes[0, j].plot(t, waves[c], color=COLORS[c], linewidth=0.5)
            axes[0, j].set_ylim(-amp, amp)
            axes[0, j].set_title(f"{LABELS[c]}  ({len(waves[c]) / SAMPLE_RATE:.2f} s)")
            image = axes[1, j].imshow(mels[c].T, origin="lower", aspect="auto", cmap="magma", vmin=lo, vmax=hi,
                                      extent=[0, mels[c].shape[0] * 160 / SAMPLE_RATE, 0, 63])
            axes[1, j].grid(False)
            axes[1, j].set_xlabel("Time (s)")
            axes[1, j].set_yticks([0, 16, 32, 48, 63])
            axes[1, j].set_yticklabels([f"{centers[i] / 1000:.1f}" for i in (0, 16, 32, 48, 63)])
        axes[0, 0].set_ylabel("Amplitude (full scale = 1)")
        axes[1, 0].set_ylabel("Frequency (kHz, mel axis)")
        axes[0, 0].set_xlim(0, span)
        fig.colorbar(image, cax=fig.add_axes([0.925, 0.2, 0.012, 0.3]), label="ln mel power")
        fig.suptitle(f"Same sentence, three sources - LibriSpeech stem {stem} (shared amplitude and colour scale)", fontsize=11)
        rms = {c: 20 * np.log10(np.sqrt((waves[c].astype(np.float64) ** 2).mean())) for c in COMPONENTS}
        hf = {c: float(np.mean(mels[c][:, 48:]) - np.mean(mels[c][:, :16])) for c in COMPONENTS}
        lines = (f"RMS level (dBFS) genuine {rms['genuine']:.1f}, MeloTTS {rms['melo']:.1f}, VITS {rms['vits']:.1f}; mean ln-power of the top 16 mel bins relative to the bottom 16: "
                 f"{hf['genuine']:.1f}, {hf['melo']:.1f}, {hf['vits']:.1f}.",
                 "Panels share amplitude and colour scales, so visible differences are real differences in these three recordings; one sentence is an illustration, the population statistics are in figures 04 and 05.")
        fig.subplots_adjust(left=0.06, right=0.9, top=0.9, bottom=0.2, wspace=0.12, hspace=0.15)
        results.append(_save(fig, plt, out_dir, f"{FIGURE_NAMES[2]}_{stem}", lines))
    return results


def fig_level(inp, out_dir, plt):
    frame = inp["frame"]
    panels = [("rms_dbfs", "Overall RMS (dBFS)"), ("noise_floor_dbfs", "Noise floor, 10th percentile frame (dBFS)"),
              ("silence_fraction", "Share of low-energy frames"), ("leading_silence_s", "Leading silence (s)"),
              ("trailing_silence_s", "Trailing silence (s)"), ("clip_fraction", "Share of samples at full scale")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    for ax, (col, title) in zip(axes.ravel(), panels):
        data = [frame.loc[frame["component"] == c, col] for c in COMPONENTS]
        ax.boxplot(data, tick_labels=[c.capitalize() for c in COMPONENTS], showfliers=False, patch_artist=True, boxprops={"alpha": 0.7})
        for patch, c in zip(ax.patches, COMPONENTS):
            patch.set_facecolor(COLORS[c])
        ax.set_title(title, fontsize=10)
    sep = {col: separation(frame, col) for col, _ in panels}
    best = max(sep, key=sep.get)
    names = dict(panels)
    clip = frame.groupby("component")["clip_fraction"].apply(lambda s: float((s > 0).mean()))
    lines = (f"Median RMS (dBFS): genuine {_median(frame, 'rms_dbfs', 'genuine'):.1f}, MeloTTS {_median(frame, 'rms_dbfs', 'melo'):.1f}, VITS {_median(frame, 'rms_dbfs', 'vits'):.1f}; "
             f"median trailing silence (s): {_median(frame, 'trailing_silence_s', 'genuine'):.2f}, {_median(frame, 'trailing_silence_s', 'melo'):.2f}, {_median(frame, 'trailing_silence_s', 'vits'):.2f}; "
             f"recordings with any full-scale sample: {clip['genuine']:.1%}, {clip['melo']:.1%}, {clip['vits']:.1%}.",
             f"Strongest single separator of genuine vs synthetic here is '{names[best]}' (AUC {sep[best]:.2f}, {verdict(sep[best])}); "
             "all six columns go to the R2-S3 shortcut probe. Low-energy rule: per-recording two-class energy threshold (Otsu), 10 dB minimum gap.")
    fig.tight_layout(rect=FOOT)
    return _save(fig, plt, out_dir, FIGURE_NAMES[3], lines), {"separation_auc": sep}


def fig_spectral(inp, out_dir, plt):
    import librosa
    frame, mel = inp["frame"], inp["mel_mean"]
    centers = librosa.mel_frequencies(n_mels=66, fmin=0.0, fmax=SAMPLE_RATE / 2)[1:-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    for c in COMPONENTS:
        m = mel[(frame["component"] == c).to_numpy()]
        q1, med, q3 = np.percentile(m, [25, 50, 75], axis=0)
        a.plot(centers / 1000, med, color=COLORS[c], linewidth=2, label=LABELS[c])
        a.fill_between(centers / 1000, q1, q3, color=COLORS[c], alpha=0.18)
    a.set_xlabel("Frequency (kHz)")
    a.set_ylabel("Mean ln mel power per recording")
    a.set_title("Average spectrum (median and quartile band across recordings)")
    a.legend(frameon=False)
    b.boxplot([frame.loc[frame["component"] == c, "hf_ratio_4k"] for c in COMPONENTS], tick_labels=[c.capitalize() for c in COMPONENTS], showfliers=False,
              patch_artist=True, boxprops={"alpha": 0.7})
    for patch, c in zip(b.patches, COMPONENTS):
        patch.set_facecolor(COLORS[c])
    b.set_ylabel("Share of STFT power at or above 4 kHz")
    b.set_title("High-band energy share")
    v4, v7 = separation(frame, "hf_ratio_4k"), separation(frame, "hf_ratio_7k")
    lines = (f"Median share of power at or above 4 kHz: genuine {_median(frame, 'hf_ratio_4k', 'genuine'):.4f}, MeloTTS {_median(frame, 'hf_ratio_4k', 'melo'):.4f}, VITS {_median(frame, 'hf_ratio_4k', 'vits'):.4f}.",
             f"The 4 kHz share separates genuine from synthetic with AUC {v4:.2f} ({verdict(v4)}); the 7 kHz share with AUC {v7:.2f} ({verdict(v7)}). Spectral balance is a legitimate cue, but the probe in R2-S3 must also test it.")
    fig.tight_layout(rect=FOOT)
    return _save(fig, plt, out_dir, FIGURE_NAMES[4], lines), {"hf4_auc": v4, "hf7_auc": v7}


def fig_mfcc(inp, out_dir, plt):
    path = inp["features_path"]
    if not path.exists():
        raise EdaError(f"{path} not found: MFCC distributions use the R2-S1 train features")
    with np.load(path, allow_pickle=False) as z:
        names, X, comp, y = [str(n) for n in z["feature_names"]], z["X"], z["component"], z["y"]
    cols = [i for i, n in enumerate(names) if n.startswith("mfcc_") and n.endswith("_mean") and n != "mfcc_00_mean"]
    comp = np.array(comp).astype(str)
    fig, ax = plt.subplots(figsize=(14, 5.6))
    width = 0.26
    for k, c in enumerate(COMPONENTS):
        data = [X[comp == c, i] for i in cols]
        pos = np.arange(len(cols)) + (k - 1) * width
        bp = ax.boxplot(data, positions=pos, widths=width * 0.9, showfliers=False, patch_artist=True, boxprops={"alpha": 0.75},
                        medianprops={"color": "black"})
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[c])
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([names[i].split("_")[1].lstrip("0") or "0" for i in cols])
    ax.set_xlabel("MFCC index (per-crop mean over a 4 s crop; C0 omitted, it tracks loudness)")
    ax.set_ylabel("MFCC mean value")
    ax.set_title(f"MFCC distributions in the B-train crop bank ({len(y):,} crops)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLORS[c], label=LABELS[c]) for c in COMPONENTS], frameon=False, ncol=3)
    seps = []
    for i in cols:
        value = auc(X[y == 0, i], X[y == 1, i])
        seps.append((max(value, 1 - value), names[i]))
    seps.sort(reverse=True)
    lines = (f"Largest genuine-vs-synthetic separation among MFCC means: {seps[0][1]} (AUC {seps[0][0]:.2f}), then {seps[1][1]} ({seps[1][0]:.2f}) and {seps[2][1]} ({seps[2][0]:.2f}).",
             "Boxes show quartiles of per-crop MFCC means for each generator; the more the three boxes overlap for a coefficient, the weaker that coefficient is as a cue.")
    fig.tight_layout(rect=FOOT)
    return _save(fig, plt, out_dir, FIGURE_NAMES[5], lines), {"top_mfcc": seps[:3]}


def fig_format(inp, out_dir, plt):
    frame = inp["frame"]
    wanted = {(c, s) for c, s in zip(frame["component"], frame["stem"])}
    table = {}
    with open(inp["headers_path"], encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["component"], stem_from_path(row["path"]))
            if key in wanted:
                table.setdefault(row["component"], {}).setdefault((row["format"], row["subtype"], row["sample_rate"], row["channels"]), 0)
                table[row["component"]][(row["format"], row["subtype"], row["sample_rate"], row["channels"])] += 1
    covered = sum(sum(v.values()) for v in table.values())
    if covered != len(frame):
        raise EdaError(f"header inspection covers {covered} of {len(frame)} B-train recordings")
    rows = [[LABELS[c], f"{fmt} / {sub}", f"{rate:,} Hz", str(ch), f"{n:,}"] for c in COMPONENTS for (fmt, sub, rate, ch), n in sorted(table[c].items())]
    fig, ax = plt.subplots(figsize=(10, 3.6 + 0.3 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["Source", "Container / codec", "Sample rate", "Channels", "Recordings"], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    ax.set_title("Source audio format of the B-train recordings (file headers read at admission)")
    single = len({k for v in table.values() for k in v}) == 1
    only = sorted({k for v in table.values() for k in v})
    lines = (f"All {covered:,} B-train recordings were read from header inspection: {len({k for v in table.values() for k in v})} distinct format combination(s) across the three sources.",
             ("Every file is identical in container, codec, sample rate and channel count, so format cannot separate the classes; resampling was never needed."
              if single and only[0][2] == SAMPLE_RATE and only[0][3] == 1 else "Formats differ between sources: this is a possible shortcut and must be carried to R2-S3."))
    fig.tight_layout(rect=FOOT)
    return _save(fig, plt, out_dir, FIGURE_NAMES[6], lines), {"format_combinations": len(only), "identical": single}


def fig_speakers(inp, out_dir, plt):
    frame = inp["frame"]
    per_speaker = inp["admission"]["per_speaker_counts"]
    speakers = sorted(set(frame["speaker_id"]))
    admitted = np.array([per_speaker[s] for s in speakers])
    sampled = frame.groupby("speaker_id").size().reindex(speakers).to_numpy()
    stems = frame.groupby("speaker_id")["stem"].nunique().reindex(speakers).to_numpy()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 5.2))
    a.hist(admitted, bins=24, color="#56B4E9")
    a.set_xlabel("Admitted utterances per speaker (all 5 sources present)")
    a.set_ylabel("Training speakers")
    a.set_title(f"What each of the {len(speakers)} training speakers could offer")
    values, counts = np.unique(sampled, return_counts=True)
    b.bar(values, counts, width=0.8, color="#D55E00")
    b.set_xlim(values.min() - 5, values.max() + 5)
    b.set_ylim(0, counts.max() * 1.15)
    b.bar_label(b.containers[0], fmt="{:,.0f} speakers")
    b.set_xticks(values if len(values) <= 12 else b.get_xticks())
    b.set_xlabel("Recordings in B train per speaker (stems x 3 sources)")
    b.set_ylabel("Training speakers")
    b.set_title("What the sampled cohort uses")
    lines = (f"{len(speakers)} training speakers; admitted utterances per speaker range {admitted.min()}-{admitted.max()} (median {int(np.median(admitted))}), "
             f"while the cohort samples {int(stems.min())}-{int(stems.max())} stems per speaker, i.e. {int(sampled.min())}-{int(sampled.max())} recordings.",
             ("Every speaker contributes the same number of recordings, so none dominates the training set; " if sampled.min() == sampled.max() else "Speakers contribute unequal numbers of recordings; ")
             + "grouped splits keep each speaker in one partition only.")
    fig.tight_layout(rect=FOOT)
    return _save(fig, plt, out_dir, FIGURE_NAMES[7], lines), {"speakers": len(speakers)}


def exclusion_rows(inp) -> list[dict]:
    adm, mat, frame = inp["admission"], inp["materialize"], inp["frame"]
    rows = [{"stage": "Admission (R1-S1)", "item": "LibriSpeech stems with all five sources present", "count": adm["stems_admitted"], "note": f"{adm['speakers']} speakers, {adm['chapters']} chapters"},
            {"stage": "Admission (R1-S1)", "item": "Stems excluded", "count": sum(adm["stems_excluded"].values()) if adm["stems_excluded"] else 0,
             "note": "header or pairing failures: none recorded" if not adm["stems_excluded"] else str(adm["stems_excluded"])},
            {"stage": "Materialise (R1-S3)", "item": "Recordings in the sampled cohort", "count": mat["recordings"], "note": f"{mat['recordings'] // 5:,} stems x 5 sources"}]
    for reason, ids in sorted(mat["excluded_ids"].items()):
        rows.append({"stage": "Materialise (R1-S3)", "item": f"Excluded at decode: {reason}", "count": len(ids), "note": ", ".join(ids)})
    for path in inp["repair"]:
        rep = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"stage": "Repair (R1-S3)", "item": "Excluded recordings repaired", "count": len(rep["repaired"]),
                     "note": f"{', '.join(rep['repaired'])}; still failing: {len(rep['still_failing'])}; retained {rep['retained_before']} -> {rep['retained_after']}"})
    rows.append({"stage": "B train (R2-S1/S2)", "item": "Recordings used in this EDA", "count": int(len(frame)),
                 "note": "unretained recordings skipped: " + str(inp["manifest"]["counts"].get("unretained_skipped", 0))})
    return rows


def fig_exclusions(inp, out_dir, plt):
    rows = exclusion_rows(inp)
    with open(out_dir / "exclusions_table.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "item", "count", "note"])
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(13, 1.8 + 0.55 * len(rows)))
    ax.axis("off")
    cells = [[r["stage"], r["item"], f"{r['count']:,}", r["note"]] for r in rows]
    tbl = ax.table(cellText=cells, colLabels=["Stage", "Item", "Count", "Note"], loc="center", cellLoc="left", colWidths=[0.16, 0.34, 0.08, 0.42])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.9)
    ax.set_title("Exclusions from admission to the B-train partition")
    dropped = sum(r["count"] for r in rows if r["item"].startswith("Excluded at decode"))
    repaired = sum(r["count"] for r in rows if r["stage"].startswith("Repair"))
    failed_ids = [i for ids in inp["materialize"]["excluded_ids"].values() for i in ids]
    in_train = sum(1 for i in failed_ids if i.split(":")[1] in COMPONENTS)
    failed_sources = sorted({i.split(":")[1] for i in failed_ids})
    lines = (f"No stems were excluded at admission; {dropped} recording{'s' if dropped != 1 else ''} ({', '.join(failed_sources) or 'none'}) failed decoding at materialisation and {repaired} {'was' if repaired == 1 else 'were'} repaired afterwards, leaving all {inp['materialize']['recordings']:,} cohort recordings retained.",
             (f"{in_train} of them belong to a B-train source, so the training partition lost {'nothing' if not in_train else 'recordings'}; ")
             + "the table is also saved as exclusions_table.csv.")
    fig.tight_layout(rect=FOOT)
    return _save(fig, plt, out_dir, FIGURE_NAMES[8], lines), {"rows": len(rows)}


def make_all(root, out_dir=None, data_dir=None) -> dict:
    """Build every figure family; returns the report dict (also written as interpretations.json)."""
    plt = _plt()
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else root / "reports" / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    inp = load_inputs(root, data_dir)
    figures, stats = [], {}
    for fn in (fig_counts, fig_duration, fig_level, fig_spectral, fig_mfcc, fig_format, fig_speakers, fig_exclusions):
        record, extra = fn(inp, out_dir, plt)
        figures.append(record)
        stats[fn.__name__] = extra
    figures.extend(fig_paired(inp, out_dir, plt))
    figures.sort(key=lambda f: f["figure"])
    files = sorted(p.name for p in out_dir.iterdir() if p.suffix in (".png", ".svg", ".csv"))
    report = {"task": "R2-S2", "scope": "Protocol B train partition only; no ChatterboxTTS, no validation or test audio",
              "plan_sha256": inp["manifest"]["plan_sha256"], "recordings": int(len(inp["frame"])),
              "measurement_settings": SETTINGS, "figure_families": len(FIGURE_NAMES), "figures": figures, "statistics": stats,
              "files": {name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest() for name in files}}
    (out_dir / "interpretations.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=float), encoding="utf-8")
    return report
