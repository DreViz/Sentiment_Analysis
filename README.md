# Sentiment Analysis

A Streamlit app that performs sentiment analysis on text and Amazon product reviews using a RoBERTa model (`cardiffnlp/twitter-roberta-base-sentiment`). Negative reviews trigger an email alert.

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Configuration

Email alert credentials are read from environment variables (see `.env.example`). Never commit real credentials to the repository.
