import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load Dataset
df = pd.read_csv(
    "spam.csv",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Convert labels
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english")

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = MultinomialNB()

model.fit(X_train, y_train)

# UI
st.title("📧 Spam Mail Detector")

message = st.text_area("Enter SMS Message")

if st.button("Check"):

    message_vector = vectorizer.transform([message])

    prediction = model.predict(message_vector)

    if prediction[0] == 1:
        st.error("🚫 Spam Message")
    else:
        st.success("✅ Ham Message")