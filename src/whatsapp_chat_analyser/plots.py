from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from src.whatsapp_chat_analyser.text_features import iter_words


def save_messages_per_sender(df: pd.DataFrame, out_png: Path, top_n: int = 20) -> None:
    base = df.loc[(~df["is_system"]) & (df["sender"].notna())].copy()
    counts = base["sender"].value_counts().head(top_n).sort_values()

    plt.figure(figsize=(10, max(4, 0.35 * len(counts))))
    sns.set_style("whitegrid")
    ax = counts.plot(kind="barh")
    ax.set_title(f"Top {len(counts)} senders by message count")
    ax.set_xlabel("Messages")
    ax.set_ylabel("Sender")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_messages_over_time(df: pd.DataFrame, out_png: Path, freq: str = "D") -> None:
    base = df.loc[~df["is_system"]].copy()
    ts = pd.to_datetime(base["datetime"], errors="coerce").dropna()
    if ts.empty:
        return
    s = ts.dt.floor(freq).value_counts().sort_index()

    plt.figure(figsize=(12, 4))
    sns.set_style("whitegrid")
    ax = s.plot()
    ax.set_title(f"Messages over time ({freq})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Messages")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_wordcloud(df: pd.DataFrame, out_png: Path, max_words: int = 200) -> None:
    base = df.loc[(~df["is_system"]) & (df["message_type"] == "text")].copy()
    words = []
    for msg in base["message"].fillna("").astype(str):
        words.extend(list(iter_words(msg)))
    if not words:
        return

    text = " ".join(words)
    wc = WordCloud(width=1400, height=800, background_color="white", max_words=max_words)
    img = wc.generate(text)

    plt.figure(figsize=(12, 6))
    plt.imshow(img, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()

