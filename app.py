import streamlit as st
import numpy as np
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

model=load_model("hate_speech_model.h5")

lemmatizer=WordNetLemmatizer()
stop_words=set(stopwords.words("english"))
vocab_size=1000
sentence_length=20

def clean_text(text):
    text=re.sub('[^a-z A-Z 0-9-]+','',text)                     # special chars
    text=" ".join([w for w in text.split() if w not in stop_words])  # stopwords
    text=re.sub(r'(http|https|ftp|ssh)://\S+','',text)             # remove URLs
    text=BeautifulSoup(text,'html.parser').get_text()               # remove HTML
    text=" ".join(text.split())                                      # remove extra spaces
    return text

def lemmatize(text):
    return " ".join(lemmatizer.lemmatize(x) for x in text.split())

def encode_text(text):
    one_hot_rep=one_hot(text,vocab_size)
    padded=pad_sequences([one_hot_rep],padding='pre',maxlen=sentence_length)
    return padded

st.title("Hate Speech/Offensive Detection App")
st.write("Enter any text below to classify it as Hate Speech,Offensive or Neutral.")

user_input=st.text_area("Type your text here:")

if st.button("Predict"):
    if len(user_input.strip())==0:
        st.warning("Please enter a valid text.")
    else:
        # preprocessing
        cleaned=clean_text(user_input)
        lem=lemmatize(cleaned)
        encoded=encode_text(lem)

        # model prediction
        pred=np.argmax(model.predict(encoded))

        # label map (modify based on your dataset classes)
        label_map={
            0: "Hate Speech",
            1: "Offensive",
            2: "Neither"
        }

        st.subheader("Prediction:")
        st.success(label_map.get(pred, "Unknown"))

        st.write("**Processed Text:**", lem)