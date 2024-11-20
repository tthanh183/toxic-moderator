from flask import Flask, request, jsonify
import numpy as np
import pickle
from langdetect import detect
from tensorflow.keras.models import load_model
from underthesea import word_tokenize
import re
import os

app = Flask(__name__)

model_dir = os.path.join(os.getcwd(), "models")

with open(os.path.join(model_dir, "tfidf_vectorizer.pkl"), 'rb') as f:
    english_vect = pickle.load(f)

english_model = load_model(os.path.join(model_dir, "toxic_comment_model.h5"))

with open(os.path.join(model_dir, "vietnamese_tfidf_vectorizer.pkl"), 'rb') as f:
    vietnamese_vect = pickle.load(f)

vietnamese_model = load_model(os.path.join(model_dir, "vietnamese_tfidf_vectorizer.h5")) 

def clean_text_english(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    return text

def clean_text_vietnamese(text):
    text = text.lower()
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = word_tokenize(text) 
    text = ' '.join(text)
    return text

def predict_toxicity(text):
    try:
        lang = detect(text)
        print(f"Detected language: {lang}")

        if lang == 'vi':  
            cleaned_text = clean_text_vietnamese(text)
            text_vector = vietnamese_vect.transform([cleaned_text])
            model = vietnamese_model
        else: 
            cleaned_text = clean_text_english(text)
            text_vector = english_vect.transform([cleaned_text])
            model = english_model

        prediction = model.predict(text_vector.toarray())

        if prediction >= 0.5:
            return "Toxic"
        else:
            return "Non-Toxic"

    except Exception as e:
        return f"Error: {e}"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data.get('text')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = predict_toxicity(text)
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)