"""EchoProof demo application (R2-N1 LR integration). Run from the repository root: streamlit run app/main.py"""
import hashlib
import os
from pathlib import Path

import streamlit as st

from echoproof.app_support import diagnostics, present, result_json, result_json_with_diagnostics
from echoproof.inference import analyze, load_inference_bundle
from echoproof import viz

st.set_page_config(page_title="EchoProof", page_icon=None, layout="centered")

DEFAULT_BUNDLE_PATH = Path("runs/r2/r2_m1_20261106T140611Z/model")
# Set ECHOPROOF_BUNDLE to a bundle folder (for example bundles/r3_n1_wavlm_head_B_calibrated) to use another model; the default is unchanged.
BUNDLE_PATH = Path(os.environ.get("ECHOPROOF_BUNDLE", str(DEFAULT_BUNDLE_PATH)))


@st.cache_resource
def load_bundle(bundle_path=None):
    """Load the configured bundle (the Protocol B LR bundle unless ECHOPROOF_BUNDLE says otherwise)."""
    target = Path(bundle_path) if bundle_path else BUNDLE_PATH
    return load_inference_bundle(target)


try:
    bundle = load_bundle()
except Exception as exc:
    st.title("EchoProof")
    st.error(f"Failed to load production model bundle: {exc}")
    st.stop()
st.title("EchoProof")
st.caption("Checks whether a speech recording shows signs of text-to-speech synthesis.")
if bundle.is_mock:
    st.error("MOCK MODEL. No trained model is connected yet. Scores below are placeholders derived from the audio hash and "
             "say nothing about whether a recording is synthetic.")

upload = st.file_uploader("Audio file (WAV, FLAC, OGG or MP3; up to 25 MiB and 60 seconds)", type=["wav", "flac", "ogg", "mp3"])
if upload is None:
    st.info("Choose a recording to begin.")
    st.stop()

if upload is not None:
    data = upload.getvalue()
    key = hashlib.sha256(data).hexdigest()
    st.audio(data)

    if st.button("Analyze", type="primary"):
        with st.spinner("Analysing"):
            st.session_state["analysis"] = (key, analyze(data, bundle))

    stored = st.session_state.get("analysis")
    if not stored or stored[0] != key:
        st.stop()                                            # a different file was chosen: the old result is not shown for it
    if stored and stored[0] == key:
        analysis = stored[1]
        view = present(analysis.result)
        {"success": st.success, "warning": st.warning, "error": st.error}[view["level"]](view["headline"])
        for line in view["lines"]:
            st.write(line)
        for warning in analysis.result.warnings:
            st.warning(warning)
        if analysis.result.status != "success":
            st.download_button("Download result (JSON)", result_json(analysis.result), file_name="echoproof_result.json", mime="application/json")
            st.stop()

        if analysis.result.status == "success":
            wave = analysis.waveform
            try:
                fig = viz.waveform_figure(wave.samples, wave.sample_rate, windows=analysis.windows)
                st.pyplot(fig)
                viz.close(fig)
                fig = viz.logmel_figure(wave.samples, wave.sample_rate)
                st.pyplot(fig)
                viz.close(fig)
            except RuntimeError as exc:
                st.warning(str(exc))
            st.subheader("Chunk scores")
            st.caption("Each 4 s window (2 s stride) is scored separately; the recording score is their mean.")
            st.dataframe({"start_s": [a / wave.sample_rate for a, _ in analysis.windows], "end_s": [b / wave.sample_rate for _, b in analysis.windows],
                          "score": [round(s, 4) for s in analysis.chunk_scores]}, hide_index=True)
            with st.expander("Score details"):
                st.json(diagnostics(analysis, bundle))
            st.download_button("Download result (JSON)", result_json_with_diagnostics(analysis, bundle), file_name="echoproof_result.json", mime="application/json")
