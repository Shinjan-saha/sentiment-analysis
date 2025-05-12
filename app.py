import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


model = tf.keras.models.load_model('models/sentiment_model.h5')


with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


label_decoder = {0: 'Negative', 1: 'Positive'}


max_length = 100


st.title("🎬 Sentiment Analysis on Movie Reviews")
st.write("Enter a movie review below and find out whether it's Positive or Negative.")

user_input = st.text_area("Your Review:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        
        input_sequence = tokenizer.texts_to_sequences([user_input])
        input_padded = pad_sequences(input_sequence, maxlen=max_length, padding='post', truncating='post')

       
        prediction = model.predict(input_padded)[0][0]
        predicted_class = int(prediction > 0.5)
        sentiment = label_decoder[predicted_class]

        st.subheader("Prediction:")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {prediction:.2f}")
