import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from nltk .tokenize import PunktTokenizer

import nltk

nltk.download('stopwords')

nltk.download('punkt_tab')
# Initialize PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

# Load the vectorizer and model from pickle files
tf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Streamlit UI
st.title('Email/SMS Spam Classifier Made by Pritiyax Shukla')

input_sms = st.text_input("Enter the message")

# Button to trigger prediction
if st.button("Predict"):
    # 1. Preprocessing
    transform_sms = transform_text(input_sms)

    # 2. Vectorization
    vector_input = tf.transform([transform_sms])

    # 3. Prediction
    result = model.predict(vector_input)[0]

    # 4. Display the result
    if result == 1:
        st.header('Spam')
    else:
        st.header("Not Spam")
