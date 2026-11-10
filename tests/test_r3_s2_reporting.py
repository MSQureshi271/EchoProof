"""R3-S2 reporting helpers: template rendering, fact consistency, figures, and a guard that the generated draft matches its template and sources."""
import hashlib
import json
import re
from pathlib import Path

import pytest

from echoproof.reporting import facts as F
from echoproof.reporting import figures as G
from echoproof.reporting import render as R

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "reports" / "final"
HAVE_PROJECT = all((ROOT / p).exists() for p in (
    "reports/r1/split_report.json", "reports/r1/admission.json", "data/manifests/cohort_v1.parquet",
    "data/eda/r2_s2_B_train/recording_stats_B_train_v1.parquet", "reports/r3/spec_v2", "reports/final/pages_3_6_template.md",
    "reports/final/pages_3_6_draft.md", "reports/final/pages_3_6_facts.json"))
needs_project = pytest.mark.skipif(not HAVE_PROJECT, reason="the project's evidence files are not present")

FACTS = {"a.one": {"value": 1, "text": "1,000", "source": "x.json", "how": "read"}, "a.two": {"value": 2, "text": "2.5", "source": "y.json", "how": "read"},
         "a.unused": {"value": 3, "text": "3", "source": "z.json", "how": "read"}}


def _body(template: str) -> str:
    return re.sub(r"\A<!--.*?-->\s*", "", template, count=1, flags=re.S)


# ---- rendering --------------------------------------------------------------------------------------------------------------------------

def test_render_fills_placeholders_and_reports_use():
    text, used = R.render("Total {{a.one}} and {{ a.two }} again {{a.one}}.", FACTS)
    assert text == "Total 1,000 and 2.5 again 1,000."
    assert used == {"a.one", "a.two"}
    assert R.unused_facts(FACTS, used) == ["a.unused"]


def test_render_refuses_an_unknown_placeholder():
    with pytest.raises(F.FactError, match="unknown facts"):
        R.render("Value {{a.one}} and {{a.typo}}", FACTS)


def test_text_without_placeholders_is_unchanged():
    assert R.render("Plain 12 text, no braces.", FACTS)[0] == "Plain 12 text, no braces."


def test_number_formatters():
    assert F.n(17550) == "17,550" and F.n(0) == "0" and F.f(0.8049, 3) == "0.805" and F.f(20.0, 1) == "20.0"


def test_disagreeing_sources_raise():
    F._agree("same", 3, 3)
    with pytest.raises(F.FactError, match="disagree"):
        F._agree("speakers", 251, 250)


def test_duplicate_fact_keys_are_refused():
    d = {}
    F._fact(d, "k", 1, "1", "s")
    with pytest.raises(F.FactError):
        F._fact(d, "k", 2, "2", "s")


# ---- figures (synthetic split report, no project data) ----------------------------------------------------------------------------------

def _synthetic_root(tmp_path):
    (tmp_path / "reports" / "r1").mkdir(parents=True)

    def sup(**kw):
        return {k: {"recordings": v, "speakers": 5} for k, v in kw.items()}

    report = {"speakers": 10, "speaker_counts": {"train/-": 6, "validation/selection": 1, "validation/calibration": 1, "validation/operating": 1, "test/-": 1},
              "protocols": {
                  "B": {"recordings": 100, "support": sup(**{"train|-|genuine|genuine": 30, "train|-|known_family|tts": 60,
                                                              "validation|selection|genuine|genuine": 10, "validation|selection|known_family|tts": 20,
                                                              "validation|selection|unseen_validation|tts": 10, "test|-|genuine|genuine": 10,
                                                              "test|-|known_family|tts": 20, "test|-|unseen_final|tts": 10})},
                  "A": {"recordings": 150, "support": sup(**{"train|-|genuine|genuine": 30, "train|-|known_family|tts": 120,
                                                              "validation|selection|genuine|genuine": 10, "validation|selection|known_family|tts": 40,
                                                              "test|-|genuine|genuine": 10, "test|-|known_family|tts": 40})}}}
    (tmp_path / "reports" / "r1" / "split_report.json").write_text(json.dumps(report), encoding="utf-8")
    return tmp_path


def test_split_figure_is_written_and_deterministic(tmp_path):
    root = _synthetic_root(tmp_path)
    a = G.split_composition(root, tmp_path / "a")
    b = G.split_composition(root, tmp_path / "b")
    assert (tmp_path / "a.png").stat().st_size > 5000
    assert (tmp_path / "a.svg").read_bytes() == (tmp_path / "b.svg").read_bytes()
    assert a["summary"]["B"]["recordings"] == 100 and len(a["interpretation"]) == 2 and b["files"]


def test_split_figure_shows_every_number(tmp_path):
    root = _synthetic_root(tmp_path)
    G.split_composition(root, tmp_path / "f")
    svg = (tmp_path / "f.svg").read_text(encoding="utf-8")
    for expected in ("100 recordings", "150 recordings", "30 + 60", "30 + 120", "10 + 20 + 10"):
        assert expected in svg, expected


# ---- the real project: facts, template and the generated draft -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def project_facts():
    return F.compute_facts(ROOT)


@needs_project
def test_every_fact_has_text_and_an_existing_source(project_facts):
    assert len(project_facts) > 100
    for key, entry in project_facts.items():
        assert entry["text"] != "" and entry["source"] and entry["how"], key
        assert (ROOT / entry["source"]).exists(), (key, entry["source"])


@needs_project
def test_template_uses_only_known_facts_and_leaves_no_placeholder(project_facts):
    template = (FINAL / "pages_3_6_template.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
    text, used = R.render(_body(template), project_facts)
    assert len(used) > 100
    assert "{{" not in text and "}}" not in text


@needs_project
def test_generated_draft_matches_template_and_sources(project_facts):
    """Fails if the draft was edited by hand, or if the template or any source changed without regenerating (`python scripts/r3_s2_report.py all`)."""
    template = (FINAL / "pages_3_6_template.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
    text, _ = R.render(_body(template), project_facts)
    draft = (FINAL / "pages_3_6_draft.md").read_bytes().decode("utf-8")
    assert draft.split("\n\n", 1)[1] == text
    record = json.loads((FINAL / "pages_3_6_facts.json").read_text(encoding="utf-8"))
    assert record["draft_sha256"] == hashlib.sha256(draft.encode("utf-8")).hexdigest()
    assert record["template_sha256"] == hashlib.sha256(template.encode("utf-8")).hexdigest()


@needs_project
def test_key_numbers_match_independent_reads_of_the_sources(project_facts):
    split = json.loads((ROOT / "reports/r1/split_report.json").read_text(encoding="utf-8"))
    assert project_facts["split.B.recordings"]["value"] == split["protocols"]["B"]["recordings"] == 12530
    assert project_facts["cohort.recordings"]["value"] == 17550
    assert project_facts["leak.violations"]["value"] == json.loads((ROOT / "reports/r1/leakage.json").read_text(encoding="utf-8"))["violation_count"] == 0
    evidence = json.loads(sorted((ROOT / "reports/r3/spec_v2").glob("r3_s1_colab_*.json"))[-1].read_text(encoding="utf-8"))
    assert project_facts["corr.hash"]["text"] == evidence["report"]["spec"]["spec_sha256"][:8]
