from __future__ import annotations

import argparse
from pathlib import Path

from whatsapp_chat_analyser.pipeline import run_full_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="WhatsApp chat EDA + topics + sentiment")
    parser.add_argument("--chat", required=True, help="Path to exported WhatsApp .txt")
    parser.add_argument("--out", required=True, help="Output directory (will be created)")
    parser.add_argument("--n_topics", type=int, default=8, help="Number of topics for LDA")
    parser.add_argument("--max_words", type=int, default=60, help="Top words to include in report")
    args = parser.parse_args()

    chat_path = Path(args.chat).expanduser()
    out_dir = Path(args.out).expanduser()

    run_full_analysis(
        chat_path=chat_path,
        out_dir=out_dir,
        n_topics=args.n_topics,
        max_words=args.max_words,
    )


if __name__ == "__main__":
    main()

