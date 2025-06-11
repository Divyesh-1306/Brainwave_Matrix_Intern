import streamlit as st
import joblib
import numpy as np
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
def predict(text):
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]
    return prediction
def main():
    st.title("Fake News Detector")
    text = st.text_area("Enter Text to Check", "")
    if st.button("Predict"):
        prediction = predict(text)
        if prediction == 0:
            st.write("Prediction: False")
        else:
            st.write("Prediction: True")

if __name__ == '__main__':
    main()
