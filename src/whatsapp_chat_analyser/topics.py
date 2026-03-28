from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import re

from src.whatsapp_chat_analyser.text_features import normalize_text
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_Other_STOPWORDS = {
    "edited","just", "like", "i'm", "you're", "he's", "it's", "i'll", "i've", "i'd",
    "yeah", "ok", "okay", "yes", "no", "lol", "haha", "hahaha", "😂", "😭"
}

# Combine English + custom stopwords
STOP_WORDS = ENGLISH_STOP_WORDS.union(_Other_STOPWORDS)

# Optional: Add basic French + Arabic stopwords
FRENCH_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "et", "ou", "mais", "dans", "sur", 
    "avec", "pour", "par", "est", "sont", "était", "je", "tu", "il", "elle", 
    "nous", "vous", "ils", "elles", "ça", "c'est", "qui", "que", "quoi", "où"
}

ARABIC_STOPWORDS = {
    "في", "من", "على", "إلى", "عن", "مع", "كان", "يكون", "هذا", "هذه", "ذلك",
    "التي", "الذي", "ما", "لا", "لم", "لن", "إن", "أن", "قال", "يقول", "كل",
    "بعض", "هناك", "هنا", "يوم", "وقت","chi","rah","mn","la","شي","واحد","3la",
    "ولا","غير","ila","أَن","مع","gha","wla","wach","3la","راه","واش","عافاك","li","لي"
}

# Final combined stop words
ALL_STOP_WORDS = STOP_WORDS.union(FRENCH_STOPWORDS).union(ARABIC_STOPWORDS)


@dataclass(frozen=True)
class TopicModelResult:
    model: LatentDirichletAllocation
    vectorizer: CountVectorizer
    topic_keywords: dict[int, list[str]]
    doc_topic: np.ndarray


def normalize_text(text: str) -> str:
    """Basic text normalization - improve this if you have a better one"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)      # remove URLs
    text = re.sub(r'[^a-zA-Z\u0600-\u06FF\s]', ' ', text)  # keep Arabic + Latin letters
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def fit_lda_topics(
    texts: Iterable[str],
    *,
    n_topics: int = 8,
    max_features: int = 5000,
    min_df: int = 3,
    max_df: float = 0.7,
    random_state: int = 42,
) -> TopicModelResult:
    
    # Keep original indices by using a list of (original_index, text) pairs
    valid_texts = []
    original_indices = []
    
    for i, t in enumerate(texts):
        if isinstance(t, str) and len(t.strip()) > 5:   # minimum length filter
            valid_texts.append(t)
            original_indices.append(i)
    
    if len(valid_texts) < max(20, n_topics * 5):
        # Too few valid messages for LDA
        return None  # we'll handle this in attach_topics

    cleaned = [normalize_text(t) for t in valid_texts]

    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),
        stop_words=list(ALL_STOP_WORDS),
        token_pattern=r"(?u)\b\w\w+\b",
    )
    
    X = vectorizer.fit_transform(cleaned)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="batch",
        max_iter=10,
    )
    doc_topic = lda.fit_transform(X)

    vocab = np.array(vectorizer.get_feature_names_out())
    topic_keywords: dict[int, list[str]] = {}
    
    for k, topic in enumerate(lda.components_):
        top_idx = np.argsort(topic)[::-1][:12]
        topic_keywords[k] = vocab[top_idx].tolist()

    return TopicModelResult(
        model=lda,
        vectorizer=vectorizer,
        topic_keywords=topic_keywords,
        doc_topic=doc_topic,
    ), original_indices   # ← return indices too


def attach_topics(df: pd.DataFrame, *, n_topics: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Select only text messages that are not system messages
    base = df.loc[(~df["is_system"]) & (df["message_type"] == "text")].copy()
    texts = base["message"].fillna("").astype(str).tolist()

    if len(texts) < max(30, n_topics * 10):
        df2 = df.copy()
        df2["topic_id"] = pd.NA
        df2["topic_score"] = pd.NA
        topics_df = pd.DataFrame(columns=["topic_id", "keywords"])
        return df2, topics_df

    # Fit LDA and get mapping of which rows were actually used
    result = fit_lda_topics(texts, n_topics=n_topics)
    
    if result is None:
        # Fallback if too few valid texts
        df2 = df.copy()
        df2["topic_id"] = pd.NA
        df2["topic_score"] = pd.NA
        topics_df = pd.DataFrame(columns=["topic_id", "keywords"])
        return df2, topics_df

    res, valid_indices = result
    
    topic_id = res.doc_topic.argmax(axis=1)
    topic_score = res.doc_topic.max(axis=1)

    # Create Series with original base indices (so lengths match)
    topic_id_series = pd.Series(pd.NA, index=base.index)
    topic_score_series = pd.Series(pd.NA, index=base.index)

    # Assign only to the rows that were used in LDA
    topic_id_series.iloc[valid_indices] = topic_id
    topic_score_series.iloc[valid_indices] = topic_score

    # Add columns safely
    base = base.assign(topic_id=topic_id_series, topic_score=topic_score_series)

    # Merge back to full dataframe
    df2 = df.copy()
    df2 = df2.merge(
        base[["datetime", "sender", "message", "topic_id", "topic_score"]],
        on=["datetime", "sender", "message"],
        how="left",
    )

    # Topics summary
    topics_df = pd.DataFrame(
        [{"topic_id": k, "keywords": " | ".join(v)} 
         for k, v in res.topic_keywords.items()]
    ).sort_values("topic_id")

    return df2, topics_df