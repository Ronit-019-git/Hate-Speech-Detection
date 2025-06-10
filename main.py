import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
from nltk.corpus import stopwords
import string

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

df = pd.read_csv("twitter_data.csv")

df['labels'] = df['class'].map({
    0: "Hate Speech Detected",
    1: "Offensive Language Detected",
    2: "No hate and offensive speech detected"
})

df = df.dropna(subset=['labels'])
df = df[['tweet', 'labels']]
df['tweet'] = df['tweet'].apply(clean)

x = np.array(df["tweet"])
y = np.array(df["labels"])

cv = CountVectorizer()
x = cv.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

test_data = "You ARe awesome"
df = cv.transform([test_data]).toarray()
print(clf.predict(df))
