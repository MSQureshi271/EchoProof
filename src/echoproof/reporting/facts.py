"""R3-S2: every number the report cites about the data, computed from the project's own evidence files.

`compute_facts(root)` returns `{key: {"value", "text", "source", "how"}}`. `text` is what the template inserts; `source` names the file; `how` says whether
the value was read or recomputed. Nothing here is typed in by hand. Where two sources describe the same quantity (for example the cohort manifest and the
split report), the function checks that they agree and raises `FactError` if they do not, so a stale or edited file cannot slip into the report.

Sources (all relative to the project root):
  reports/r1/admission.json, split_report.json, materialize.json, repair_*.json, leakage.json
  reports/phase2/p2_08-20261103T150049Z-dd446277f234/reconciliation.json
  data/manifests/cohort_v1.parquet                       (recomputed counts, hours, medians, hashes)
  data/eda/r2_s2_B_train/recording_stats_B_train_v1.parquet   (recomputed EDA statistics)
  reports/eda/interpretations.json                       (EDA figure captions; computed numbers are cross-checked against them)
  reports/r2/r2_s3_shortcut_probe.json                   (shortcut probe)
  src/echoproof/data/corruption.py, data/chunks.py, data/leakage.py   (corruption specification, preprocessing hash, leakage check list)
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

RECONCILIATION = "reports/phase2/p2_08-20261103T150049Z-dd446277f234/reconciliation.json"
COMPONENT_LABEL = {"genuine": "Genuine", "melo": "MeloTTS", "vits": "VITS", "xtts": "XTTS v2", "chatterbox": "ChatterboxTTS"}


class FactError(ValueError):
    """Two sources disagree, or a required source is missing."""


def n(value) -> str:
    return f"{int(round(value)):,}"


def f(value, digits=2) -> str:
    return f"{value:.{digits}f}"


def _fact(facts, key, value, text, source, how="read"):
    if key in facts:
        raise FactError(f"duplicate fact {key}")
    facts[key] = {"value": value, "text": text, "source": source, "how": how}


def _load(root: Path, rel: str):
    path = root / rel
    if not path.exists():
        raise FactError(f"missing source: {rel}")
    return json.loads(path.read_text(encoding="utf-8"))


def _agree(name, a, b):
    if a != b:
        raise FactError(f"sources disagree on {name}: {a!r} vs {b!r}")


def _ci(lo_hi, digits=3) -> str:
    return f"[{lo_hi[0]:.{digits}f}, {lo_hi[1]:.{digits}f}]"


def compute_facts(root) -> dict:
    root = Path(root)
    facts: dict = {}

    # ---- admission and source (Phase 2 / R1-S1) -------------------------------------------------------------------------------
    adm = _load(root, "reports/r1/admission.json")
    rec = _load(root, RECONCILIATION)
    _fact(facts, "data.stems_total", adm["stems_admitted"], n(adm["stems_admitted"]), "reports/r1/admission.json")
    _fact(facts, "data.speakers", adm["speakers"], n(adm["speakers"]), "reports/r1/admission.json")
    _fact(facts, "data.chapters", adm["chapters"], n(adm["chapters"]), "reports/r1/admission.json")
    _fact(facts, "data.utt_min", adm["utterances_per_speaker"]["min"], n(adm["utterances_per_speaker"]["min"]), "reports/r1/admission.json")
    _fact(facts, "data.utt_median", adm["utterances_per_speaker"]["median"], n(adm["utterances_per_speaker"]["median"]), "reports/r1/admission.json")
    _fact(facts, "data.utt_max", adm["utterances_per_speaker"]["max"], n(adm["utterances_per_speaker"]["max"]), "reports/r1/admission.json")
    _fact(facts, "data.stems_excluded", len(adm["stems_excluded"]), n(len(adm["stems_excluded"])), "reports/r1/admission.json")
    if not adm["format_consistent"] or adm["format_errors"] or adm["chapters_shared_by_multiple_speakers"]:
        raise FactError("admission report records a format problem or a chapter shared by speakers; the page text would be wrong")
    comps = adm["components"]
    _fact(facts, "data.components", len(comps), n(len(comps)), "reports/r1/admission.json")
    stages = rec["stages"]
    ok = {c: v["ok"] for c, v in stages["header_inspected"]["counts"].items()}
    _agree("header-inspected files per component", set(ok.values()), {adm["stems_admitted"]})
    _fact(facts, "data.audio_files", sum(ok.values()), n(sum(ok.values())), RECONCILIATION)
    _fact(facts, "data.audio_gb", sum(stages["hashed_inventory"]["bytes"][c] for c in ok) / 1e9,
          f(sum(stages["hashed_inventory"]["bytes"][c] for c in ok) / 1e9, 1), RECONCILIATION, "recomputed from the byte inventory")
    _fact(facts, "data.sample_decode_ok", stages["sample_decoded"]["counts"]["ok"], n(stages["sample_decoded"]["counts"]["ok"]), RECONCILIATION)
    _fact(facts, "data.relationship_candidate_edges", rec["pairing"]["candidate_edges"], n(rec["pairing"]["candidate_edges"]), RECONCILIATION)
    _fact(facts, "data.relationship_verified_edges", rec["pairing"]["verified_edges"], n(rec["pairing"]["verified_edges"]), RECONCILIATION)
    _fact(facts, "data.dedup_state", stages["deduplicated"]["state"], stages["deduplicated"]["state"].replace("_", " "), RECONCILIATION)
    _fact(facts, "data.archive_comparison", stages["downloaded"]["independent_archive_comparison"],
          stages["downloaded"]["independent_archive_comparison"].replace("_", " "), RECONCILIATION)
    _fact(facts, "data.language_state", rec["components"]["genuine"]["language"], rec["components"]["genuine"]["language"], RECONCILIATION)

    # ---- cohort manifest (recomputed) --------------------------------------------------------------------------------------------
    import pandas as pd
    from echoproof.data.cohort import read_table
    cohort_path = root / "data/manifests/cohort_v1.parquet"
    if not cohort_path.exists():
        raise FactError("missing source: data/manifests/cohort_v1.parquet")
    cohort = pd.DataFrame(read_table(str(cohort_path)))
    src = "data/manifests/cohort_v1.parquet"
    _fact(facts, "cohort.recordings", len(cohort), n(len(cohort)), src, "recomputed")
    _fact(facts, "cohort.stems", cohort.stem.nunique(), n(cohort.stem.nunique()), src, "recomputed")
    _fact(facts, "cohort.speakers", cohort.speaker_id.nunique(), n(cohort.speaker_id.nunique()), src, "recomputed")
    _agree("cohort speakers vs admission speakers", cohort.speaker_id.nunique(), adm["speakers"])
    _fact(facts, "cohort.hours", cohort.duration_s.sum() / 3600, f(cohort.duration_s.sum() / 3600), src, "recomputed")
    rates = sorted(set(cohort.original_sample_rate))
    if rates != [16000]:
        raise FactError(f"cohort sample rates are {rates}; the page text says every file is 16 kHz")
    _fact(facts, "cohort.sample_rate_khz", 16, "16", src, "recomputed")
    _fact(facts, "cohort.retained", int(cohort.retained.sum()), n(cohort.retained.sum()), src, "recomputed")
    _fact(facts, "cohort.distinct_pcm_hashes", cohort.pcm_sha256.nunique(), n(cohort.pcm_sha256.nunique()), src, "recomputed")
    for comp in COMPONENT_LABEL:
        c = cohort[cohort.component == comp]
        _fact(facts, f"cohort.{comp}.recordings", len(c), n(len(c)), src, "recomputed")
        _fact(facts, f"cohort.{comp}.hours", c.duration_s.sum() / 3600, f(c.duration_s.sum() / 3600), src, "recomputed")
        _fact(facts, f"cohort.{comp}.median_s", float(c.duration_s.median()), f(float(c.duration_s.median()), 1), src, "recomputed")
    _fact(facts, "cohort.file_sha256", hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
          hashlib.sha256(cohort_path.read_bytes()).hexdigest()[:8], src, "recomputed")

    # ---- split report ---------------------------------------------------------------------------------------------------------------
    sp = _load(root, "reports/r1/split_report.json")
    sps = "reports/r1/split_report.json"
    _agree("planned stems", sp["planned_stems"], cohort.stem.nunique())
    _agree("split-report speakers", sp["speakers"], adm["speakers"])
    _fact(facts, "split.per_speaker_train", sp["per_speaker"]["train"], n(sp["per_speaker"]["train"]), sps)
    _fact(facts, "split.per_speaker_validation", sp["per_speaker"]["validation"], n(sp["per_speaker"]["validation"]), sps)
    _fact(facts, "split.per_speaker_test", sp["per_speaker"]["test"], n(sp["per_speaker"]["test"]), sps)
    sc = sp["speaker_counts"]
    val_total = sc["validation/selection"] + sc["validation/calibration"] + sc["validation/operating"]
    _agree("speaker total", sc["train/-"] + val_total + sc["test/-"], sp["speakers"])
    for key, value in (("train", sc["train/-"]), ("validation", val_total), ("test", sc["test/-"]),
                       ("selection", sc["validation/selection"]), ("calibration", sc["validation/calibration"]), ("operating", sc["validation/operating"])):
        _fact(facts, f"split.speakers_{key}", value, n(value), sps)
    _fact(facts, "split.seed", sp["seed"], n(sp["seed"]), sps)
    for proto in ("A", "B"):
        _fact(facts, f"split.{proto}.recordings", sp["protocols"][proto]["recordings"], n(sp["protocols"][proto]["recordings"]), sps)
        _fact(facts, f"split.{proto}.hash", sp["protocols"][proto]["split_sha256"], sp["protocols"][proto]["split_sha256"][:8], sps)
    supB = sp["protocols"]["B"]["support"]
    supA = sp["protocols"]["A"]["support"]

    def total(support, part, role=None, cls=None):
        return sum(v["recordings"] for k, v in support.items()
                   if k.split("|")[0] == part and (role is None or k.split("|")[1] == role) and (cls is None or k.split("|")[2] == cls))

    _agree("protocol B recordings", total(supB, "train") + total(supB, "validation") + total(supB, "test"), sp["protocols"]["B"]["recordings"])
    _agree("protocol A recordings", total(supA, "train") + total(supA, "validation") + total(supA, "test"), sp["protocols"]["A"]["recordings"])
    for tag, support in (("A", supA), ("B", supB)):
        _fact(facts, f"split.{tag}.train", total(support, "train"), n(total(support, "train")), sps, "recomputed")
        _fact(facts, f"split.{tag}.test", total(support, "test"), n(total(support, "test")), sps, "recomputed")
        for role in ("selection", "calibration", "operating"):
            _fact(facts, f"split.{tag}.{role}", total(support, "validation", role), n(total(support, "validation", role)), sps, "recomputed")
    _fact(facts, "split.B.train_genuine", total(supB, "train", None, "genuine"), n(total(supB, "train", None, "genuine")), sps, "recomputed")
    _fact(facts, "split.B.train_known", total(supB, "train", None, "known_family"), n(total(supB, "train", None, "known_family")), sps, "recomputed")
    _fact(facts, "split.B.test_genuine", total(supB, "test", None, "genuine"), n(total(supB, "test", None, "genuine")), sps, "recomputed")
    _fact(facts, "split.B.test_known", total(supB, "test", None, "known_family"), n(total(supB, "test", None, "known_family")), sps, "recomputed")
    _fact(facts, "split.B.test_unseen", total(supB, "test", None, "unseen_final"), n(total(supB, "test", None, "unseen_final")), sps, "recomputed")
    _fact(facts, "split.B.validation_unseen", total(supB, "validation", None, "unseen_validation"),
          n(total(supB, "validation", None, "unseen_validation")), sps, "recomputed")
    _fact(facts, "split.A.test_tts", total(supA, "test", None, "known_family"), n(total(supA, "test", None, "known_family")), sps, "recomputed")

    # ---- materialisation, repair, chunks ------------------------------------------------------------------------------------------
    mat = _load(root, "reports/r1/materialize.json")
    repair_files = sorted((root / "reports/r1").glob("repair_*.json"))
    if len(repair_files) != 1:
        raise FactError(f"expected exactly one repair report, found {len(repair_files)}")
    rep = json.loads(repair_files[0].read_text(encoding="utf-8"))
    _agree("materialised recordings", mat["recordings"], len(cohort))
    _fact(facts, "clean.shard_gb", mat["shard_gb"], f(mat["shard_gb"]), "reports/r1/materialize.json")
    _fact(facts, "clean.decode_failures", sum(mat["exclusions"].values()), n(sum(mat["exclusions"].values())), "reports/r1/materialize.json")
    _fact(facts, "clean.decode_failure_id", mat["excluded_ids"]["decode_error:OSError"][0], mat["excluded_ids"]["decode_error:OSError"][0],
          "reports/r1/materialize.json")
    _fact(facts, "clean.repaired", len(rep["repaired"]), n(len(rep["repaired"])), repair_files[0].relative_to(root).as_posix())
    _agree("retained after repair", rep["retained_after"], int(cohort.retained.sum()))
    _fact(facts, "clean.still_failing", len(rep["still_failing"]), n(len(rep["still_failing"])), repair_files[0].relative_to(root).as_posix())
    _fact(facts, "clean.chunks", rep["chunks_after"], n(rep["chunks_after"]), repair_files[0].relative_to(root).as_posix())
    from echoproof.data import chunks as chunks_mod
    _agree("preprocessing hash in materialize.json vs code", mat["preprocessing_hash"], chunks_mod.PREPROCESSING_HASH)
    _fact(facts, "chunks.window_s", chunks_mod.WINDOW_SECONDS, n(chunks_mod.WINDOW_SECONDS), "src/echoproof/data/chunks.py")
    _fact(facts, "chunks.stride_s", chunks_mod.STRIDE_SECONDS, n(chunks_mod.STRIDE_SECONDS), "src/echoproof/data/chunks.py")
    _fact(facts, "chunks.preprocessing_hash", chunks_mod.PREPROCESSING_HASH, chunks_mod.PREPROCESSING_HASH[:8], "src/echoproof/data/chunks.py")

    # ---- leakage ----------------------------------------------------------------------------------------------------------------------
    lk = _load(root, "reports/r1/leakage.json")
    lks = "reports/r1/leakage.json"
    if not lk["passed"] or lk["violation_count"] != 0:
        raise FactError("the leakage report records violations; the page text claims none")
    _fact(facts, "leak.violations", lk["violation_count"], n(lk["violation_count"]), lks)
    for proto in ("A", "B"):
        _fact(facts, f"leak.{proto}.assignments", lk["summary"][proto]["assignments"], n(lk["summary"][proto]["assignments"]), lks)
        _fact(facts, f"leak.{proto}.chunks", lk["summary"][proto]["chunks"], n(lk["summary"][proto]["chunks"]), lks)
        _agree(f"protocol {proto} assignments", lk["summary"][proto]["assignments"], sp["protocols"][proto]["recordings"])
    _agree("chunk total", lk["summary"]["A"]["chunks"] + lk["summary"]["B"]["chunks"], rep["chunks_after"])
    leak_src = (root / "src/echoproof/data/leakage.py").read_text(encoding="utf-8")
    kinds = sorted(set(re.findall(r'_violation\("([a-z_]+)"', leak_src)))
    _fact(facts, "leak.check_kinds", len(kinds), n(len(kinds)), "src/echoproof/data/leakage.py", "counted from the source")

    # ---- EDA (B train, recomputed) ----------------------------------------------------------------------------------------------------
    stats_path = root / "data/eda/r2_s2_B_train/recording_stats_B_train_v1.parquet"
    if not stats_path.exists():
        raise FactError("missing source: data/eda/r2_s2_B_train/recording_stats_B_train_v1.parquet")
    stats = pd.read_parquet(stats_path)
    inter = _load(root, "reports/eda/interpretations.json")
    from echoproof.eda.figures import separation
    es = "data/eda/r2_s2_B_train/recording_stats_B_train_v1.parquet"
    comps_train = ["genuine", "melo", "vits"]
    _agree("B-train components", sorted(stats.component.unique()), sorted(comps_train))
    _fact(facts, "eda.train_recordings", len(stats), n(len(stats)), es, "recomputed")
    for c in comps_train:
        d = stats[stats.component == c]
        _fact(facts, f"eda.{c}.recordings", len(d), n(len(d)), es, "recomputed")
        _fact(facts, f"eda.{c}.median_s", float(d.duration_s.median()), f(float(d.duration_s.median()), 1), es, "recomputed")
        _fact(facts, f"eda.{c}.short_pct", float((d.duration_s <= 4).mean() * 100), f(float((d.duration_s <= 4).mean() * 100), 1), es, "recomputed")
        _fact(facts, f"eda.{c}.rms_median", float(d.rms_dbfs.median()), f(float(d.rms_dbfs.median()), 1), es, "recomputed")
        _fact(facts, f"eda.{c}.trailing_median", float(d.trailing_silence_s.median()), f(float(d.trailing_silence_s.median()), 2), es, "recomputed")
    syn = stats[stats.label == 1]
    gen = stats[stats.label == 0]
    _fact(facts, "eda.ratio", len(syn) / len(gen), f(len(syn) / len(gen), 1), es, "recomputed")
    _fact(facts, "eda.duration_auc", separation(stats, "duration_s"), f(separation(stats, "duration_s")), es, "recomputed")
    _fact(facts, "eda.leading_auc", separation(stats, "leading_silence_s"), f(separation(stats, "leading_silence_s")), es, "recomputed")
    _fact(facts, "eda.hf4k_auc", separation(stats, "hf_ratio_4k"), f(separation(stats, "hf_ratio_4k")), es, "recomputed")
    caption = {e["figure"]: " ".join(e["interpretation"]) for e in inter["figures"]}
    _fact(facts, "eda.format_combinations", 1, "1", "reports/eda/interpretations.json")
    for key, figure in (("eda.duration_auc", "02_duration_histograms"), ("eda.leading_auc", "04_level_and_silence"), ("eda.hf4k_auc", "05_spectral_balance")):
        if facts[key]["text"] not in caption[figure]:
            raise FactError(f"{key}={facts[key]['text']} is not in the saved caption of {figure}; the EDA files are inconsistent")
    for c in comps_train:
        if facts[f"eda.{c}.median_s"]["text"] not in caption["02_duration_histograms"]:
            raise FactError(f"median duration of {c} is not in the saved caption of figure 02")
    if "1 distinct format combination" not in caption["07_format_audit"]:
        raise FactError("the format audit no longer reports a single format combination")

    # ---- shortcut probe ---------------------------------------------------------------------------------------------------------------
    pr = _load(root, "reports/r2/r2_s3_shortcut_probe.json")
    prs = "reports/r2/r2_s3_shortcut_probe.json"
    res = pr["results"]
    prim = res["feature_sets"]["primary (plan: duration, RMS, silence fraction, sample rate)"]
    _fact(facts, "probe.primary_A", prim["A"]["roc_auc"], f"{prim['A']['roc_auc']:.3f} {_ci(prim['A']['auc_ci95'])}", prs)
    _fact(facts, "probe.primary_C", prim["C"]["roc_auc"], f"{prim['C']['roc_auc']:.3f} {_ci(prim['C']['auc_ci95'])}", prs)
    allm = res["feature_sets"]["all measurements (adds high-band shares)"]
    _fact(facts, "probe.all_A", allm["A"]["roc_auc"], f"{allm['A']['roc_auc']:.3f} {_ci(allm['A']['auc_ci95'])}", prs)
    _fact(facts, "probe.all_C", allm["C"]["roc_auc"], f"{allm['C']['roc_auc']:.3f} {_ci(allm['C']['auc_ci95'])}", prs)
    sfs = res["single_features"]
    for feat, short in (("leading_silence_s", "leading"), ("silence_fraction", "silencefrac"), ("duration_s", "duration"), ("rms_dbfs", "rms")):
        _fact(facts, f"probe.{short}_A", sfs[feat]["A"]["roc_auc"], f"{sfs[feat]['A']['roc_auc']:.3f} {_ci(sfs[feat]['A']['auc_ci95'])}", prs)
        _fact(facts, f"probe.{short}_C", sfs[feat]["C"]["roc_auc"], f"{sfs[feat]['C']['roc_auc']:.3f} {_ci(sfs[feat]['C']['auc_ci95'])}", prs)
    gbm = res["supplementary"]["gradient_boosting_all_measurements"]
    _fact(facts, "probe.gbm_A", gbm["A"]["roc_auc"], f"{gbm['A']['roc_auc']:.3f} {_ci(gbm['A']['auc_ci95'])}", prs)
    pg = prim["A"]["per_generator_auc"]
    for g in ("melo", "vits", "xtts"):
        _fact(facts, f"probe.gen_{g}", pg[g], f"{pg[g]:.3f}", prs)
    m = res["matching"]
    _fact(facts, "probe.matched_selection", m["selection"]["recordings_matched"], n(m["selection"]["recordings_matched"]), prs)
    _fact(facts, "probe.full_selection", m["selection"]["recordings_full"], n(m["selection"]["recordings_full"]), prs)
    _fact(facts, "probe.duration_auc_before", 1 - m["selection"]["duration_auc_before"], f(1 - m["selection"]["duration_auc_before"]), prs,
          "folded: 1 - the stored AUC, which is below 0.5 because genuine recordings are longer")
    _fact(facts, "probe.duration_auc_after", m["selection"]["duration_auc_after"], f(m["selection"]["duration_auc_after"]), prs)
    rev = sorted(k for k, v in sfs.items() if v["A"].get("reversed"))
    _fact(facts, "probe.reversed", rev, ", ".join(rev), prs)
    survive = pr["conclusions"]["single_features_that_survive_duration_matching"]
    _fact(facts, "probe.survive", survive, ", ".join(survive), prs)
    _agree("probe split hash vs split report", pr["inputs"]["split_sha256"], sp["protocols"]["B"]["split_sha256"])

    # ---- corruption specification (code) -------------------------------------------------------------------------------------------
    from echoproof.data import corruption
    cs = "src/echoproof/data/corruption.py"
    _fact(facts, "corr.version", corruption.SPEC_VERSION, n(corruption.SPEC_VERSION), cs)
    _fact(facts, "corr.hash", corruption.SPEC_SHA256, corruption.SPEC_SHA256[:8], cs)
    _fact(facts, "corr.conditions", list(corruption.CONDITION_NAMES), ", ".join(f"`{c}`" for c in corruption.CONDITION_NAMES), cs)
    _fact(facts, "corr.seed", corruption.MASTER_SEED, n(corruption.MASTER_SEED), cs)

    # ---- corruption evidence (version 2) ------------------------------------------------------------------------------------------------
    evidence_files = sorted((root / "reports/r3/spec_v2").glob("r3_s1_colab_*.json"))
    if not evidence_files:
        raise FactError("missing source: reports/r3/spec_v2/r3_s1_colab_*.json (the version 2 Colab evidence)")
    ev = json.loads(evidence_files[-1].read_text(encoding="utf-8"))
    evs = evidence_files[-1].relative_to(root).as_posix()
    if not ev["verdict"]["verified"]:
        raise FactError("the version 2 Colab evidence is not verified; the page text says it is")
    _agree("version 2 evidence spec hash vs code", ev["report"]["spec"]["spec_sha256"], corruption.SPEC_SHA256)
    sample = ev["report"]["sample"]
    _fact(facts, "corr.sample_recordings", sample["recordings"], n(sample["recordings"]), evs)
    _fact(facts, "corr.sample_variants", sample["variants_built"], n(sample["variants_built"]), evs)
    _fact(facts, "corr.sample_speakers", sample["speakers"], n(sample["speakers"]), evs)
    if sample["variants_built"] != sample["variants_expected"] or ev["report"]["failures"]:
        raise FactError("the version 2 sample run did not build every variant")
    variants = pd.read_csv(root / "reports/r3/spec_v2/r3_s1_corruption_sample_variants.csv")
    for cond in ("noise_20", "noise_10"):
        snr = variants.loc[variants.condition == cond, "measured_snr_db"]
        if abs(snr.min() - snr.max()) > 1e-6:
            raise FactError(f"{cond} measured SNR is not constant across variants")
        _fact(facts, f"corr.{cond}_snr", float(snr.iloc[0]), f(float(snr.iloc[0]), 1), evs, "recomputed from the variants table")
    probe = _load(root, "reports/r3/r3_s1_mp3_probe_shards.json")["summary"]
    mps = "reports/r3/r3_s1_mp3_probe_shards.json"
    _fact(facts, "corr.mp3_32_snr", probe["mp3_16k_32"]["snr_gaincomp_db"]["median"], f(probe["mp3_16k_32"]["snr_gaincomp_db"]["median"], 1), mps)
    _fact(facts, "corr.mp3_128_snr", probe["mp3_16k_128"]["snr_gaincomp_db"]["median"], f(probe["mp3_16k_128"]["snr_gaincomp_db"]["median"], 1), mps)
    _fact(facts, "corr.mp3_64_snr", probe["mp3_16k_64"]["snr_gaincomp_db"]["median"], f(probe["mp3_16k_64"]["snr_gaincomp_db"]["median"], 1), mps)
    rows = pd.read_csv(root / "reports/r3/r3_s1_mp3_probe_shards_rows.csv")
    by_comp = rows[rows.candidate == "mp3_16k_32"].groupby("component").snr_gaincomp_db.median()
    rows_src = "reports/r3/r3_s1_mp3_probe_shards_rows.csv"
    _fact(facts, "corr.mp3_32_genuine", float(by_comp["genuine"]), f(float(by_comp["genuine"]), 1), rows_src, "recomputed")
    _fact(facts, "corr.mp3_32_xtts", float(by_comp["xtts"]), f(float(by_comp["xtts"]), 1), rows_src, "recomputed")
    acc = _load(root, "reports/r3/r3_s1_v2_acceptance_local.json")
    accs = "reports/r3/r3_s1_v2_acceptance_local.json"
    _agree("acceptance-run spec hash vs code", acc["specification"]["sha256"], corruption.SPEC_SHA256)
    a32 = acc["summary"]["mp3_32"]
    if a32["below_floor"] != 0:
        raise FactError("some recordings fell below the MP3 correlation floor; the page text says none did")
    _fact(facts, "corr.accept_recordings", acc["measured"], n(acc["measured"]), accs)
    _fact(facts, "corr.accept_min_corr", a32["lag0_correlation"]["min"], f(a32["lag0_correlation"]["min"], 3), accs)
    _fact(facts, "corr.accept_floor", a32["floor"], f(a32["floor"], 1), accs)

    # ---- documented test exposure (R2-U1 extraction-only exception) -------------------------------------------------------------------
    u1_files = sorted((root / "reports/r2").glob("r2_u1_verify_*.json"))
    if not u1_files:
        raise FactError("missing source: reports/r2/r2_u1_verify_*.json")
    u1 = json.loads(u1_files[-1].read_text(encoding="utf-8"))
    u1s = u1_files[-1].relative_to(root).as_posix()
    by_component = u1["counts"]["by_component"]
    if "chatterbox" in by_component:
        raise FactError("the WavLM cache contains ChatterboxTTS embeddings; the page text says it does not")
    if not u1["include_test_genuine"]:
        raise FactError("the WavLM cache was built without test genuine recordings; the page text describes that exception")
    _fact(facts, "exposure.test_chunks", u1["counts"]["by_partition"]["test"], n(u1["counts"]["by_partition"]["test"]), u1s)

    # ---- validation-set class ratios and saved figure captions ----------------------------------------------------------------------
    for part, key in (("validation", "val"), ("test", "test")):
        genuine = total(supB, part, None, "genuine")
        synthetic = sum(v["recordings"] for k, v in supB.items() if k.split("|")[0] == part and k.split("|")[2] != "genuine")
        _fact(facts, f"split.B.ratio_{key}", synthetic / genuine, f(synthetic / genuine, 1), sps, "recomputed")
    figures = {e["figure"]: e["interpretation"] for e in inter["figures"]}
    for key, figure in (("fig01", "01_class_generator_counts"), ("fig02", "02_duration_histograms"), ("fig04", "04_level_and_silence"),
                        ("fig05", "05_spectral_balance"), ("fig06", "06_mfcc_distributions"), ("fig07", "07_format_audit"), ("fig08", "08_speaker_group_sizes")):
        _fact(facts, f"cap.{key}", figures[figure], " ".join(figures[figure]), "reports/eda/interpretations.json")
    _fact(facts, "cap.probe", pr["figure"]["interpretation"], " ".join(pr["figure"]["interpretation"]), prs)
    return facts


def write_facts(facts: dict, path) -> None:
    Path(path).write_text(json.dumps(facts, indent=2, sort_keys=True, default=str), encoding="utf-8")
