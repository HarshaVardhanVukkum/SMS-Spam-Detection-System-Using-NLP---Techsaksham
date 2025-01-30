import streamlit as st
import joblib
import re

# Load the trained model and vectorizer
model = joblib.load('spam_model.pkl')  # Replace with your model path
vectorizer = joblib.load('vectorizer.pkl')  # Replace with your vectorizer path

# Function to clean input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Function to predict if the message is spam or ham
def predict_sms(message):
    cleaned_message = clean_text(message)
    message_vector = vectorizer.transform([cleaned_message])
    prediction = model.predict(message_vector)
    return prediction[0]

# Streamlit app UI
st.title("SMS Spam Detection")

st.write("Enter a message to check if it's spam or ham:")

# User input
user_message = st.text_input("Enter your message")

# Prediction on button click
if st.button("Predict"):
    if user_message:
        prediction = predict_sms(user_message)
        if prediction == 'spam':
            st.write("❌ This message is **Spam**!")
        else:
            st.write("✅ This message is **Ham** (Not Spam).")
    else:
        st.write("Please enter a message to predict.")
