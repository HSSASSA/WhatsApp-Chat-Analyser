from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\S+")
_NONWORD_EDGE_RE = re.compile(r"(^[^\w]+|[^\w]+$)", re.UNICODE)


_Other_STOPWORDS = {
    "edited","just", "like", "i'm", "you're", "he's", "it's", "i'll", "i've", "i'd",
    "yeah", "ok", "okay", "yes", "no", "lol", "haha", "hahaha", "😂", "😭"
}

STOP_WORDS = ENGLISH_STOP_WORDS.union(_Other_STOPWORDS)

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

def normalize_text(s: str) -> str:
    s = s.strip()
    s = _URL_RE.sub(" ", s)
    s = _MENTION_RE.sub(" ", s)
    s = s.replace("\u200e", "").replace("\u200f", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def iter_words(text: str) -> Iterable[str]:
    """
    Unicode-friendly word extraction:
    - keeps letters across languages
    - removes punctuation/edges
    """
    text = normalize_text(text).lower()
    for token in text.split():
        token = _NONWORD_EDGE_RE.sub("", token)
        if not token:
            continue
        # drop pure numbers / underscores
        if re.fullmatch(r"[\d_]+", token):
            continue
        yield token


def top_words(
    df: pd.DataFrame,
    *,
    n: int = 50,
    per_sender: bool = False,
    min_len: int = 2,
) -> pd.DataFrame:
    stop = set(ALL_STOP_WORDS)

    base = df.loc[(~df["is_system"]) & (df["message_type"] == "text")].copy()

    def _count(series: pd.Series) -> Counter:
        c: Counter = Counter()
        for msg in series.dropna().astype(str):
            for w in iter_words(msg):
                if len(w) < min_len:
                    continue
                if w in stop:
                    continue
                c[w] += 1
        return c

    if not per_sender:
        c = _count(base["message"])
        items = c.most_common(n)
        return pd.DataFrame(items, columns=["word", "count"])

    out_rows = []
    for sender, g in base.groupby("sender", dropna=False):
        c = _count(g["message"])
        for w, cnt in c.most_common(n):
            out_rows.append({"sender": sender, "word": w, "count": cnt})
    return pd.DataFrame(out_rows)

