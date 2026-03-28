from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from dateutil import parser as dateutil_parser


_LINE_RE = re.compile(
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s+"
    r"(?P<time>\d{1,2}:\d{2})"
    r"(?:\s*(?P<ampm>[AP]M))?"
    r"\s+-\s+"
    r"(?P<rest>.*)$"
)


@dataclass(frozen=True)
class ParsedRow:
    dt: Optional[datetime]
    sender: Optional[str]
    message: str
    is_system: bool
    message_type: str  # text | media | deleted | waiting | system


def _classify_message(text: str, is_system: bool) -> str:
    t = text.strip()
    if is_system:
        return "system"
    if t in ("<Media omitted>",):
        return "media"
    if t == "This message was deleted":
        return "deleted"
    if t == "Waiting for this message":
        return "waiting"
    return "text"


def _split_sender_and_message(rest: str) -> tuple[Optional[str], str, bool]:
    """
    WhatsApp export line after ' - ' is usually:
      - "Name: message"
      - system event with no ": " (created group, joined, left, etc.)
    """
    if ": " in rest:
        sender, _, msg = rest.partition(": ")
        sender = sender.strip()
        # Some exports use empty sender for certain events; treat that as system.
        if sender:
            return sender, msg, False
    return None, rest.strip(), True


def _parse_datetime(date_s: str, time_s: str, ampm: Optional[str]) -> Optional[datetime]:
    # Handles both 2-digit and 4-digit years, and optional AM/PM.
    raw = f"{date_s} {time_s}{(' ' + ampm) if ampm else ''}".strip()
    try:
        return dateutil_parser.parse(raw, dayfirst=False, yearfirst=False)
    except Exception:
        return None


def iter_parsed_rows(lines: Iterable[str]) -> Iterable[ParsedRow]:
    current: Optional[ParsedRow] = None

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        m = _LINE_RE.match(line)
        if not m:
            # Continuation of previous message (multi-line content).
            if current is not None:
                current = ParsedRow(
                    dt=current.dt,
                    sender=current.sender,
                    message=current.message + "\n" + line,
                    is_system=current.is_system,
                    message_type=current.message_type,
                )
            continue

        # New message begins: emit previous
        if current is not None:
            yield current

        dt = _parse_datetime(m.group("date"), m.group("time"), m.group("ampm"))
        sender, msg, is_system = _split_sender_and_message(m.group("rest"))
        message_type = _classify_message(msg, is_system=is_system)
        current = ParsedRow(
            dt=dt,
            sender=sender,
            message=msg,
            is_system=is_system,
            message_type=message_type,
        )

    if current is not None:
        yield current


def parse_chat_txt(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        rows = list(iter_parsed_rows(f))

    df = pd.DataFrame(
        {
            "datetime": [r.dt for r in rows],
            "sender": [r.sender for r in rows],
            "message": [r.message for r in rows],
            "is_system": [r.is_system for r in rows],
            "message_type": [r.message_type for r in rows],
        }
    )

    # Derived fields for easier analysis.
    df["date"] = pd.to_datetime(df["datetime"]).dt.date
    df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
    df["weekday"] = pd.to_datetime(df["datetime"]).dt.day_name()
    df["message_len"] = df["message"].fillna("").astype(str).str.len()
    df["word_count"] = df["message"].fillna("").astype(str).str.split().str.len()
    return df

