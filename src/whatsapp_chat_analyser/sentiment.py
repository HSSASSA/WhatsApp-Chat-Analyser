from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.whatsapp_chat_analyser.text_features import normalize_text


@dataclass(frozen=True)
class SentimentBackend:
    name: str


def _try_transformers_backend() -> Optional[tuple[SentimentBackend, object]]:
    try:
        from transformers import pipeline  # type: ignore
    except Exception:
        return None

    # Multilingual sentiment model; will download weights on first run.
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    clf = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
    return SentimentBackend(name=f"transformers:{model_name}"), clf


def choose_backend() -> tuple[SentimentBackend, object]:
    t = _try_transformers_backend()
    if t is not None:
        return t
    return SentimentBackend(name="vader"), SentimentIntensityAnalyzer()


def score_sentiment(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Adds:
      - sentiment_label
      - sentiment_score
    Returns (df2, backend_name).
    """
    base = df.loc[(~df["is_system"]) & (df["message_type"] == "text")].copy()
    df2 = df.copy()

    backend, clf = choose_backend()

    if backend.name.startswith("transformers:"):
        texts = base["message"].fillna("").astype(str).map(normalize_text).tolist()
        # pipeline outputs labels like: "Negative", "Neutral", "Positive"
        out = clf(texts, truncation=True)
        labels = [o["label"].lower() for o in out]
        scores = [float(o["score"]) for o in out]
        base = base.assign(sentiment_label=labels, sentiment_score=scores)
    else:
        analyzer: SentimentIntensityAnalyzer = clf  # type: ignore[assignment]
        labels = []
        scores = []
        for msg in base["message"].fillna("").astype(str).map(normalize_text):
            vs = analyzer.polarity_scores(msg)
            compound = float(vs["compound"])
            if compound >= 0.05:
                lab = "positive"
            elif compound <= -0.05:
                lab = "negative"
            else:
                lab = "neutral"
            labels.append(lab)
            scores.append(compound)
        base = base.assign(sentiment_label=labels, sentiment_score=scores)

    df2 = df2.merge(
        base[["datetime", "sender", "message", "sentiment_label", "sentiment_score"]],
        on=["datetime", "sender", "message"],
        how="left",
    )
    return df2, backend.name

