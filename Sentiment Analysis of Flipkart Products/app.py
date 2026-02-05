from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    confidence = None


    if request.method == "POST":
        review = request.form["review_text"]

        clean_review = clean_text(review)
        vector = vectorizer.transform([clean_review])

        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector).max()

        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"

    return render_template(
        "index.html",
        sentiment=sentiment,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
