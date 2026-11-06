# Application (R1-N2 skeleton)

```
uv pip install -e ".[app,analysis]" --python .venv\Scripts\python.exe
streamlit run app/main.py
```

Run it from the repository root (the `.streamlit/config.toml` there sets the upload limit).

Until a real LR bundle exists (R2-N1) the app uses `echoproof.inference.mock_bundle()`. It shows a red banner and every result is
labelled MOCK. The mock score is a hash of the audio, not a detection result. R2-N1 replaces `load_bundle()` in `app/main.py`.

Logic that does not need Streamlit lives in `src/echoproof/` (`inference.py`, `viz.py`, `app_support.py`) and is tested in
`tests/test_r1_n2_inference.py`.
