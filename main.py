import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

#Load dataset
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

#Load pretrained model
model=load_model('simple_rnn_imdb.h5')

import re

#decode reviews 
import re

def preprocess_text(text):
    # 1. Clean the text (Crucial for RNNs)
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    
    # 2. Encode words with the +3 offset
    # We cap at 9999 to match your 10,000 max_features
    counts = [min(word_index.get(word, 2) + 3, 9999) for word in words]
    
    
    # Every IMDB review in the training set starts with index 1
    final_indices = [1] + counts
    
    # 4. Pad the sequence
    padded_review = pad_sequences([final_indices], maxlen=500)
    return padded_review

## Prediction function
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

    return sentiment,prediction[0][0]

##streamlit app
import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

#User input
user_input=st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)

    #Make prediction
    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

    #Display result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review')