import os
import streamlit as st
st.write("Debug: Current working directory listing:")
st.write(os.listdir())

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords only once
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# App title
st.title("Clinical Note Urgency Classifier")
st.write("Enter a clinical note and receive a predicted urgency level: **EMERGENCY**, **URGENT**, **NON-URGENT**, or **ROUTINE**.")

# Preprocess user text
def clean_pipe(text_list):
    cleaned = []
    for text in text_list:
        text = text.lower()
        text = re.sub(r'\b(emergency|urgent|non[-\s]?urgent|routine)\b', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        cleaned.append(' '.join(tokens))
    return cleaned

# Load and train model
@st.cache_resource
def train_model():
    df = pd.read_csv("synthetic_clinical_urgency_dataset_noisy_augmented.csv")
    df = df[['text', 'label']].dropna()

    # Clean
    cleaned_text = clean_pipe(df['text'].tolist())

    # Encode
    le = LabelEncoder()
    y = le.fit_transform(df['label'])

    # Vectorize
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=3, max_df=0.9)
    X = tfidf.fit_transform(cleaned_text)

    # Train
    clf = MultinomialNB()
    clf.fit(X, y)

    return clf, tfidf, le

model, vectorizer, label_encoder = train_model()

# Text input
user_input = st.text_area("Enter clinical note text below")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a clinical note.")
    else:
        cleaned_input = clean_pipe([user_input])
        input_vec = vectorizer.transform(cleaned_input)
        pred = model.predict(input_vec)[0]
        prob = model.predict_proba(input_vec).max()

        label = label_encoder.inverse_transform([pred])[0]
        st.success(f"**Predicted Urgency Level:** {label}")
        st.info(f"Model Confidence: {prob:.2%}")
