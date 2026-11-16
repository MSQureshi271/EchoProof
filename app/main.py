"""EchoProof demo application (R4-N2: locked production B bundle). Run from the repository root: streamlit run app/main.py

Set ECHOPROOF_BUNDLE to a bundle folder only for development; that bundle is then not verified against the lock and the app says so.
"""
import hashlib
import os

import streamlit as st

from echoproof import app_support as support
from echoproof.app_support import diagnostics, present, result_json, result_json_with_diagnostics
from echoproof.inference import analyze
from echoproof import viz

st.set_page_config(page_title="EchoProof", page_icon=None, layout="centered")

BUNDLE_OVERRIDE = os.environ.get("ECHOPROOF_BUNDLE") or None      # None means the locked production bundle


@st.cache_resource(show_spinner="Loading and verifying the model")
def load_status(bundle_path=None):
    """The only cached object: the verified model bundle. User audio and results are never cached (they live in st.session_state)."""
    return support.prepare_bundle(bundle_path)


def clear_session():
    """Forget the chosen file's result and give the uploader a fresh identity so the selected file is dropped too."""
    for name in ("analysis", "notice", "busy"):
        st.session_state.pop(name, None)
    st.session_state["uploader_n"] = st.session_state.get("uploader_n", 0) + 1


st.title("EchoProof")
st.caption("Checks whether a speech recording shows signs of text-to-speech synthesis.")

status = load_status(BUNDLE_OVERRIDE)
if not status.ok:
    load_status.clear()                                   # do not keep a failure cached; a restart or a fixed bundle is retried
    st.error(f"Model unavailable ({status.code}). {status.message}")
    st.write("This is a problem with the installed model, not with your audio. No analysis can be done until it is fixed.")
    st.stop()
bundle = status.bundle
for line in status.warnings:
    st.warning(line)
if support.score_type_warning(bundle.score_type):
    st.warning(support.UNCALIBRATED_WARNING)

st.subheader("Scope and privacy")
st.info(support.SCOPE_STATEMENT)
st.caption("A model score is evidence from one model, not proof of how a recording was made. It never identifies a person.")

upload = st.file_uploader("Audio file to analyse (WAV, FLAC, OGG or MP3; up to 25 MiB and 60 seconds)",
                          type=["wav", "flac", "ogg", "mp3"], key=f"upload_{st.session_state.get('uploader_n', 0)}")
if upload is None:
    for name in ("analysis", "notice"):                   # the file was removed or reset: nothing from it may linger
        st.session_state.pop(name, None)
    st.info("Choose a recording to begin.")
    st.stop()

data = upload.getvalue()
key = hashlib.sha256(data).hexdigest()
stored = st.session_state.get("analysis")
if stored and stored[0] != key:                           # another file was chosen: drop the old result, not just hide it
    st.session_state.pop("analysis", None)
    st.session_state.pop("notice", None)
st.audio(data)

busy = bool(st.session_state.get("busy", False)) or support.INFERENCE_GATE.busy
left, right = st.columns(2)
with left:
    clicked = st.button("Analyze recording", type="primary", disabled=busy, help="Scores the chosen recording with the locked model.")
with right:
    if st.button("Clear file and result", disabled=busy, help="Removes the chosen file and any result from this session."):
        clear_session()
        st.rerun()
if clicked:
    st.session_state["busy"] = True
    st.session_state.pop("notice", None)
    st.rerun()
if st.session_state.get("busy"):
    try:
        with st.spinner("Analysing"):
            ran, analysis_obj = support.INFERENCE_GATE.run(analyze, data, bundle)
        if ran:
            st.session_state["analysis"] = (key, analysis_obj)
        else:
            st.session_state["notice"] = support.BUSY_MESSAGE
    finally:
        st.session_state["busy"] = False
    st.rerun()
if st.session_state.get("notice"):
    st.warning(st.session_state["notice"])

stored = st.session_state.get("analysis")
if not stored or stored[0] != key:
    st.stop()

analysis = stored[1]
view = present(analysis.result)
{"success": st.success, "warning": st.warning, "error": st.error}[view["level"]](view["headline"])
st.write(support.outcome_text(analysis.result))
for line in view["lines"]:
    st.write(line)
type_warning = support.score_type_warning(analysis.result.score_type) if analysis.result.status == "success" else None
if type_warning:
    st.warning(type_warning)
for warning in analysis.result.warnings:
    st.warning(warning)
if analysis.result.status != "success":
    st.download_button("Download result (JSON)", result_json(analysis.result), file_name="echoproof_result.json", mime="application/json")
    st.stop()

wave = analysis.waveform
try:
    fig = viz.waveform_figure(wave.samples, wave.sample_rate, windows=analysis.windows)
    st.pyplot(fig)
    viz.close(fig)
    st.caption(support.waveform_alt_text(analysis))
    fig = viz.logmel_figure(wave.samples, wave.sample_rate)
    st.pyplot(fig)
    viz.close(fig)
    st.caption(support.logmel_alt_text(analysis))
except RuntimeError as exc:
    st.warning(str(exc))
st.subheader("Chunk scores")
st.caption("Each 4 s window (2 s stride) is scored separately; the recording score is their mean.")
st.dataframe({"start_s": [a / wave.sample_rate for a, _ in analysis.windows], "end_s": [b / wave.sample_rate for _, b in analysis.windows],
              "score": [round(s, 4) for s in analysis.chunk_scores]}, hide_index=True)
with st.expander("Score details and known cost of the locked policy"):
    st.write(support.policy_cost_note())
    st.json(diagnostics(analysis, bundle))
st.download_button("Download result (JSON)", result_json_with_diagnostics(analysis, bundle), file_name="echoproof_result.json", mime="application/json")
