# Hate-Speech-Detection
A simple machine learning project to detect hate speech and offensive language in tweets using natural language processing and a decision tree classifier. Includes a Flask API for easy deployment.

# Hate Speech Detection Web Application

This project is a machine learning–based web application that detects **hate speech**, **offensive language**, or **neutral content** in user-submitted tweets. It leverages natural language processing (NLP) techniques for text preprocessing and employs a `DecisionTreeClassifier` for classification. The backend is built with **Flask** and exposes a REST API for predictions.

---

## 🔍 Overview

- **Objective**: Identify and classify tweets into one of three categories:
  - `Hate Speech Detected`
  - `Offensive Language Detected`
  - `No hate and offensive speech detected`
- **Technologies Used**:
  - Python, Flask, Scikit-learn, NLTK
  - CountVectorizer for text feature extraction
  - Decision Tree Classifier for model training

---

## 📁 Dataset

The project uses a labeled dataset in CSV format: `twitter_data.csv`

| Column Name | Description                       |
|-------------|-----------------------------------|
| tweet       | The tweet text                    |
| class       | Label: 0 (Hate), 1 (Offensive), 2 (Neutral) |

The numeric `class` labels are mapped as follows:

| Class | Label                          |
|-------|--------------------------------|
| 0     | Hate Speech Detected           |
| 1     | Offensive Language Detected    |
| 2     | No hate and offensive speech detected |

---

## ⚙️ Features

- Clean and preprocess text data (removes URLs, punctuations, digits, stopwords)
- Trains a Decision Tree classifier on vectorized tweet text
- RESTful API with `/predict` endpoint for predictions
- CORS-enabled for cross-origin frontend integration
- Basic static frontend supported (`index.html`)

---

## 🚀 Getting Started

### Prerequisites

Ensure Python 3.7+ is installed. Install required packages:

```bash
pip install flask flask-cors pandas numpy scikit-learn nltk
