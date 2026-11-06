"""EchoProof demo application (R1-N2 skeleton). Run from the repository root: streamlit run app/main.py"""
import hashlib

import streamlit as st

from echoproof.app_support import present, result_json
from echoproof.inference import analyze, mock_bundle
from echoproof import viz

st.set_page_config(page_title="EchoProof", page_icon=None, layout="centered")


@st.cache_resource
def load_bundle():
    """R2-N1 replaces this with the locked LR bundle. Until then the app runs the clearly labelled mock."""
    return mock_bundle()


bundle = load_bundle()
st.title("EchoProof")
st.caption("Checks whether a speech recording shows signs of text-to-speech synthesis.")
if bundle.is_mock:
    st.error("MOCK MODEL. No trained model is connected yet. Scores below are placeholders derived from the audio hash and "
             "say nothing about whether a recording is synthetic.")

upload = st.file_uploader("Audio file (WAV, FLAC, OGG or MP3; up to 25 MiB and 60 seconds)", type=["wav", "flac", "ogg", "mp3"])
if upload is None:
    st.info("Choose a recording to begin.")
    st.stop()

data = upload.getvalue()
key = hashlib.sha256(data).hexdigest()
st.audio(data)

if st.button("Analyze", type="primary"):
    with st.spinner("Analysing"):
        st.session_state["analysis"] = (key, analyze(data, bundle))

stored = st.session_state.get("analysis")
if not stored or stored[0] != key:
    st.stop()                                            # a different file was chosen: the old result is not shown for it
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
st.download_button("Download result (JSON)", result_json(analysis.result), file_name="echoproof_result.json", mime="application/json")
