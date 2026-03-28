# WhatsApp Chat Analyser (EDA + Topics + Sentiment)

This project parses exported WhatsApp `.txt` chats and generates:
- EDA: messages per contact, activity over time, message lengths, top words
- Topic modeling (LDA) to discover discussed themes
- Sentiment analysis (VADER by default; optional multilingual transformer if installed)

## Web app (upload + results)

Run a local web app where you can upload `chat.txt` and see results:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Quick start

### 1) Install

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

### 2) Run analysis

```bash
python run_analysis.py --chat "WhatsApp Chat with الشؤون الطلابية.txt" --out "outputs/sho2on"
```

Outputs:
- `outputs/.../messages.csv`: parsed messages
- `outputs/.../topics.csv`: per-message topic assignment + topic keywords
- `outputs/.../report.md`: summary report
- `outputs/.../figures/*.png`: charts + wordcloud

## Use with other chats

Export any WhatsApp chat to `.txt` (with media omitted is fine), then run:

```bash
python run_analysis.py --chat "path/to/chat.txt" --out "outputs/chat_name"
```

## Optional: better multilingual sentiment (Arabic/French/etc.)

If you want stronger multilingual sentiment, install transformers + torch, then re-run:

```bash
python -m pip install transformers torch --upgrade
```

The analyser will automatically switch to a multilingual sentiment model when available.

## Notes
- System events (created group / joined / left / deleted / media omitted) are parsed and labeled separately.
- Multi-line messages are supported (continuation lines are appended to the previous message).

