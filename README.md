# Hate Speech Detection Using LSTM (Deep Learning NLP Project)

A complete deep-learning–based system that detects Hate Speech, Offensive Language, and Normal content from user-generated text.
This project uses advanced NLP preprocessing, a multi-layer LSTM model, and an interactive Streamlit web app for real-time predictions.

## Overview

This project builds an LSTM neural network to classify text into:

- 0 – Hate Speech

- 1 – Offensive Language

- 2 – Normal / Neither

It includes full preprocessing (cleaning, stopword removal, lemmatization), model training with stacked LSTMs, and a user-friendly Streamlit interface for testing any custom input text.

## Text Preprocessing

The raw dataset (tweets) is processed through:

- Removal of special characters

- Stopword removal (NLTK)

- URL removal

- HTML tag stripping (BeautifulSoup)

- Lowercasing and whitespace cleanup

- Word lemmatization (WordNet Lemmatizer)

- One-hot encoding

- Sequence padding (Keras)

This ensures clean, normalized input for the LSTM model.

## Model Architecture

The model is built using TensorFlow/Keras:

- Embedding layer

- LSTM(100, return_sequences=True)

- LSTM(50, return_sequences=True)

- LSTM(50)

- Dense(vocab_size, activation="softmax")

Loss: sparse categorical crossentropy
Optimizer: Adam
Labels: 0 (Hate), 1 (Offensive), 2 (Normal)

## Streamlit Web App

A fully functional Streamlit app (app.py) allows users to:

- Enter text

- View preprocessing output

- Get real-time predictions

- See whether the input is Hate Speech,         Offensive, or Normal

- Run the app using: streamlit run app.py

## Example Outputs
Input Text	                    Prediction
- “I love this!”             	    Normal
- “You are such an idiot.”	    Offensive
- “People of X group are trash.”	Hate Speech
