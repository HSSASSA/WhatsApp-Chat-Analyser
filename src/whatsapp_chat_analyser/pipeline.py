from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.whatsapp_chat_analyser.parser import parse_chat_txt
from src.whatsapp_chat_analyser.plots import (
    save_messages_over_time,
    save_messages_per_sender,
    save_wordcloud,
)
from src.whatsapp_chat_analyser.sentiment import score_sentiment
from src.whatsapp_chat_analyser.text_features import top_words
from src.whatsapp_chat_analyser.topics import attach_topics


def _write_report(
    *,
    out_path: Path,
    df: pd.DataFrame,
    backend_name: str,
    top_words_df: pd.DataFrame,
    topics_df: pd.DataFrame,
    max_words: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(df)
    system = int(df["is_system"].sum())
    msgs = total - system
    senders = int(df.loc[df["sender"].notna() & (~df["is_system"]), "sender"].nunique())

    sent_counts = (
        df.loc[(~df["is_system"]) & (df["message_type"] == "text"), "sentiment_label"]
        .value_counts(dropna=False)
        .to_dict()
    )

    lines = []
    lines.append("# WhatsApp Chat Analysis Report")
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"- **total rows**: {total}")
    lines.append(f"- **messages (non-system)**: {msgs}")
    lines.append(f"- **system events**: {system}")
    lines.append(f"- **unique senders**: {senders}")
    lines.append("")

    lines.append("## EDA highlights")
    top_senders = (
        df.loc[(~df["is_system"]) & (df["sender"].notna()), "sender"]
        .value_counts()
        .head(15)
    )
    lines.append("- **top senders (message count)**:")
    for name, cnt in top_senders.items():
        lines.append(f"  - {name}: {int(cnt)}")
    lines.append("")

    lines.append(f"## Top repeated words (top {min(max_words, len(top_words_df))})")
    for _, r in top_words_df.head(max_words).iterrows():
        lines.append(f"- {r['word']}: {int(r['count'])}")
    lines.append("")

    lines.append("## Topics (LDA)")
    if topics_df.empty:
        lines.append("- Not enough text messages to build topics reliably.")
    else:
        for _, r in topics_df.iterrows():
            lines.append(f"- **topic {int(r['topic_id'])}**: {r['keywords']}")
    lines.append("")

    lines.append("## Sentiment")
    lines.append(f"- **backend**: {backend_name}")
    if sent_counts:
        lines.append("- **label distribution**:")
        for k, v in sent_counts.items():
            lines.append(f"  - {k}: {int(v)}")
    lines.append("")

    lines.append("## Figures")
    lines.append("- `figures/messages_per_sender.png`")
    lines.append("- `figures/messages_over_time.png`")
    lines.append("- `figures/wordcloud.png`")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_full_analysis(
    *,
    chat_path: Path,
    out_dir: Path,
    n_topics: int = 8,
    max_words: int = 60,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"

    df = parse_chat_txt(chat_path)
    df.to_csv(out_dir / "messages.csv", index=False, encoding="utf-8-sig")

    df, backend_name = score_sentiment(df)

    df, topics_df = attach_topics(df, n_topics=n_topics)
    topics_df.to_csv(out_dir / "topics.csv", index=False, encoding="utf-8-sig")

    tw = top_words(df, n=max(200, max_words))
    tw.to_csv(out_dir / "top_words.csv", index=False, encoding="utf-8-sig")

    save_messages_per_sender(df, fig_dir / "messages_per_sender.png")
    save_messages_over_time(df, fig_dir / "messages_over_time.png", freq="D")
    save_wordcloud(df, fig_dir / "wordcloud.png")

    _write_report(
        out_path=out_dir / "report.md",
        df=df,
        backend_name=backend_name,
        top_words_df=tw,
        topics_df=topics_df,
        max_words=max_words,
    )

