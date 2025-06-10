from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
from nltk.corpus import stopwords
import string
import pickle
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize NLTK
try:
    stopword = set(stopwords.words("english"))
except LookupError:
    nltk.download('stopwords')
    stopword = set(stopwords.words("english"))

stemmer = nltk.SnowballStemmer("english")


def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = [word for word in text.split() if word not in stopword]
    text = " ".join(text)
    return text


# Load and prepare data
df = pd.read_csv("twitter_data.csv")
df['labels'] = df['class'].map({
    0: "Hate Speech Detected",
    1: "Offensive Language Detected",
    2: "No hate and offensive speech detected"
})
df = df.dropna(subset=['labels'])
df = df[['tweet', 'labels']]
df['tweet'] = df['tweet'].apply(clean)

# Vectorize and train model
cv = CountVectorizer()
x = cv.fit_transform(np.array(df["tweet"]))
y = np.array(df["labels"])

clf = DecisionTreeClassifier()
clf.fit(x, y)


@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    cleaned_text = clean(text)
    text_vector = cv.transform([cleaned_text])
    prediction = clf.predict(text_vector)[0]

    return jsonify({
        'text': text,
        'prediction': prediction
    })


if __name__ == '__main__':
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)