# Sentiment Analysis

A Streamlit-based web application that performs sentiment analysis on free-form text and on reviews scraped live from Amazon product pages. It uses a fine-tuned RoBERTa model and sends an automated email alert whenever a scraped review is classified as negative.

## Overview

The app exposes a simple Streamlit UI with two sections:

- **Analyze Text** — submit any text snippet and get an instant sentiment label (Negative / Neutral / Positive) with a confidence score.
- **URL of product** — submit an Amazon product URL; the app scrapes the first five reviews from the page, classifies each one, and emails an alert for every review classified as negative.

Classification runs locally via Hugging Face `transformers` + PyTorch using the `cardiffnlp/twitter-roberta-base-sentiment` model.

## Features

- Real-time sentiment analysis of arbitrary text
- Amazon product review scraping with batch classification (top 5 reviews)
- Automatic Gmail alert for each negative review
- Confidence-scored, three-class output
- Lightweight Streamlit UI with a Lottie animation header

## How it works

1. **Input** — the user submits either a text snippet or an Amazon product URL through the Streamlit UI.
2. **Scraping** (URL path only) — `requests` fetches the product page and `BeautifulSoup` parses the HTML, extracting the text of the first five review elements.
3. **Inference** — each text is tokenized and passed through the RoBERTa sequence classifier. The logits are softmax-normalized into a probability distribution over the three classes; the argmax becomes the predicted label.
4. **Alerting** — when a scraped review is classified as `Negative`, the app composes a MIME email and sends it over an authenticated SMTP (STARTTLS) connection to the configured recipient.

## Tech stack

| Layer | Technology |
|------|-----------|
| Language | Python 3 |
| UI | Streamlit |
| Animation | streamlit-lottie |
| NLP model | `cardiffnlp/twitter-roberta-base-sentiment` (Hugging Face Transformers) |
| Deep learning runtime | PyTorch |
| HTTP / scraping | requests, BeautifulSoup4 |
| Email transport | smtplib + email.mime (Python standard library, SMTP with STARTTLS) |
| Configuration | python-dotenv |

## Project structure

```
Sentiment_Analysis/
├── app.py              # Streamlit entrypoint: UI, scraping, inference, alerting
├── requirements.txt    # Pinned Python dependencies
├── .env.example        # Template for required environment variables
├── .gitignore          # Excludes .env and Python artifacts
└── README.md
```

## Prerequisites

- Python 3.8+
- A Gmail account with 2-Step Verification enabled and an **App Password** generated for SMTP access (a regular account password will not work). Create one at https://myaccount.google.com/apppasswords.
- ~500 MB of free disk space for the first-time model weights download.

## Installation

```bash
# 1. Clone
git clone https://github.com/DreViz/Sentiment_Analysis.git
cd Sentiment_Analysis

# 2. (Recommended) create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials
cp .env.example .env     # on Windows PowerShell: Copy-Item .env.example .env
# then edit .env and fill in real values
```

## Configuration

Email credentials are loaded from environment variables via `python-dotenv`. Define them in a `.env` file at the project root. `.env` is gitignored and must never be committed.

| Variable | Description |
|---------|------------|
| `SENDER_EMAIL` | Gmail address used to authenticate with the SMTP server and send the alert. |
| `RECEIVER_EMAIL` | Inbox that receives the negative-review alerts. |
| `EMAIL_PASSWORD` | Gmail **App Password** for `SENDER_EMAIL` (not the account's regular password). |

## Running

```bash
streamlit run app.py
```

Streamlit prints a local URL (typically `http://localhost:8501`). Open it in your browser to use the app. The first run downloads the RoBERTa model weights from the Hugging Face Hub.

## Notes and limitations

- **Amazon page structure is fragile.** Review extraction relies on specific CSS classes (`a-section review aok-relative` for the review container, `review-text` for the body). Amazon updates these frequently; if scraping returns no reviews, the selectors likely need updating. Plain `requests.get` without browser headers may also be rate-limited or blocked.
- **Domain shift.** The model is fine-tuned on Twitter data. Applied to long-form Amazon reviews, accuracy will be lower than on short social posts.
- **Input length.** `analyze_sentiment` runs the tokenizer with `truncation=False`, so very long inputs can exceed the model's 512-token limit and raise an error. Keep inputs to a few sentences for reliable results.
- **Alert volume.** Every negative review in a scraped batch triggers a separate email; there is no batching or rate limiting.

## Security

Email credentials are read from environment variables, never from source. If you fork or clone this repository, generate your own App Password and keep it in `.env` only. If credentials are ever committed accidentally, rotate them immediately and rewrite the repository history before publishing.
