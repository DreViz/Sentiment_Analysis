import pandas as pd
import streamlit as st
from streamlit_lottie import st_lottie
import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
animation_url1 = "https://lottie.host/dd7f2ccb-f1a4-46ab-9367-ac7a766c382f/1mpGXTnenM.json"


# Load Lottie animation data from URLs
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.header('Sentiment Analysis')

# Load RoBERTa model and tokenizer
roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name)
tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
labels = ['Negative', 'Neutral', 'Positive']

# Define function for sentiment analysis using RoBERTa
def analyze_sentiment(text):
    encoded_text = tokenizer(text, return_tensors='pt', truncation=False, padding=True)
    output = model(**encoded_text)
    scores = torch.softmax(output.logits, dim=1).detach().numpy()[0]
    sentiment_index = scores.argmax()
    sentiment_label = labels[sentiment_index]
    return sentiment_label, scores[sentiment_index]

# Define function to scrape reviews from an Amazon product page
def scrape_amazon_reviews(product_url):
    response = requests.get(product_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    review_elements = soup.find_all("div", class_="a-section review aok-relative")
    reviews = []
    for review_element in review_elements[:5]: 
       
        review_text = review_element.find("span", class_="review-text").text.strip()

        reviews.append(review_text)
    
        sentiment_label, sentiment_score = analyze_sentiment(review_text)
        if sentiment_label == 'Negative':
            send_email_alert(review_text)  # Send email alert for negative review

    return reviews

# Function to send email alert for negative review
def send_email_alert(review_text):

    sender_email = "vaibhavdeori1@gmail.com"
    receiver_email = "vaibhavdeori01@gmail.com"
    password = "ejcp fpdq reek ewer"


    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Negative Review Alert"
    body = f"The following review is negative:\n\n{review_text}"
    message.attach(MIMEText(body, "plain"))

    # Establish a connection with the SMTP server
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())

# Analyze Text
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        sentiment_label, sentiment_score = analyze_sentiment(text)
        st.write('Sentiment:', sentiment_label)
        st.write('Confidence Score:', sentiment_score)


# Scrape Reviews from Amazon Product Page
with st.expander('URL of product'):
    product_url = st.text_input('Enter the link to the Amazon product page:')
    if product_url:
        reviews = scrape_amazon_reviews(product_url)
        if reviews:
            st.write('Reviews scraped successfully:')
            for review in reviews:
                sentiment_label, sentiment_score = analyze_sentiment(review)
                st.write('Review:', review)
                st.write('Sentiment:', sentiment_label)
                st.write('Confidence Score:', sentiment_score)
                st.write('---')
        else:
            st.write('No reviews found.')

st_lottie(load_lottie_url(animation_url1), speed=1, height=200, key="lottie1")    