import streamlit as st
import joblib

st.set_page_config(
    page_title="Spam Message Detection",
    page_icon="📩"
)

@st.cache_resource
def load_model():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, tfidf = load_model()

st.title("📩 Spam Message Detection System")
st.write("Enter a message to check if it is Spam or Not Spam.")

message = st.text_area("Enter your message:")

if st.button("Predict"):
    if message.strip():
        message_tfidf = tfidf.transform([message])
        prediction = model.predict(message_tfidf)[0]

        if prediction == "spam":
            st.error("🚨 Spam Message!")
        else:
            st.success("✅ Not Spam Message!")
    else:
        st.warning("Please enter a message.")
