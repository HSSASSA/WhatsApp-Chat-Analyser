from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.whatsapp_chat_analyser.pipeline import run_full_analysis


def _safe_name(name: str) -> str:
    keep = []
    for ch in name.strip():
        if ch.isalnum() or ch in ("-", "_", ".", " "):
            keep.append(ch)
    out = "".join(keep).strip().replace(" ", "_")
    return out or "chat"


def _zip_dir_bytes(dir_path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in dir_path.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(dir_path)))
    return buf.getvalue()


st.set_page_config(page_title="WhatsApp Chat Analyser", layout="wide")

st.title("WhatsApp Chat Analyser")
st.caption("Upload an exported WhatsApp .txt chat to generate EDA, topics, and sentiment.")

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("WhatsApp chat .txt", type=["txt"])
    n_topics = st.slider("Number of topics (LDA)", min_value=3, max_value=20, value=8, step=1)
    max_words = st.slider("Top repeated words to show", min_value=20, max_value=200, value=60, step=10)
    run = st.button("Run analysis", type="primary", disabled=uploaded is None)

if not uploaded:
    st.info("Upload a WhatsApp exported chat `.txt` file to begin.")
    st.stop()

base_name = _safe_name(Path(uploaded.name).stem)
out_dir = Path("outputs") / "web" / f"{base_name}_{int(time.time())}"
out_dir.mkdir(parents=True, exist_ok=True)
chat_path = out_dir / uploaded.name

if run:
    chat_path.write_bytes(uploaded.getbuffer())
    with st.status("Running analysis…", expanded=False):
        run_full_analysis(chat_path=chat_path, out_dir=out_dir, n_topics=n_topics, max_words=max_words)
    st.success("Done.")

if not (out_dir / "messages.csv").exists():
    st.warning("Click **Run analysis** to generate results.")
    st.stop()

# Load outputs
df = pd.read_csv(out_dir / "messages.csv")
top_words = pd.read_csv(out_dir / "top_words.csv") if (out_dir / "top_words.csv").exists() else pd.DataFrame()
topics = pd.read_csv(out_dir / "topics.csv") if (out_dir / "topics.csv").exists() else pd.DataFrame()
report_md = (out_dir / "report.md").read_text(encoding="utf-8") if (out_dir / "report.md").exists() else ""

colA, colB = st.columns([2, 1], gap="large")
with colA:
    st.subheader("Overview")
    st.markdown(report_md or "_No report generated._")

with colB:
    st.subheader("Download")
    zip_bytes = _zip_dir_bytes(out_dir)
    st.download_button(
        "Download results (.zip)",
        data=zip_bytes,
        file_name=f"{out_dir.name}.zip",
        mime="application/zip",
        use_container_width=True,
    )
    st.caption(f"Results folder: `{out_dir.as_posix()}`")

st.divider()
st.subheader("Figures")
fig_dir = out_dir / "figures"
fcols = st.columns(3)
paths = [
    ("Messages per sender", fig_dir / "messages_per_sender.png"),
    ("Messages over time", fig_dir / "messages_over_time.png"),
    ("Wordcloud", fig_dir / "wordcloud.png"),
]
for i, (title, p) in enumerate(paths):
    with fcols[i]:
        st.markdown(f"**{title}**")
        if p.exists():
            st.image(str(p), use_container_width=True)
        else:
            st.caption("Not available.")

st.divider()
st.subheader("Tables")
t1, t2 = st.tabs(["Messages (sample)", "Top words / Topics"])
with t1:
    st.dataframe(df.head(200), use_container_width=True)
with t2:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Top repeated words**")
        st.dataframe(top_words.head(100), use_container_width=True)
    with c2:
        st.markdown("**Topics**")
        st.dataframe(topics, use_container_width=True)

