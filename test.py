import numpy as np
import pickle
from langdetect import detect
from tensorflow.keras.models import load_model
from underthesea import word_tokenize
import re

with open('tfidf_vectorizer.pkl', 'rb') as f:
    english_vect = pickle.load(f)
english_model = load_model('toxic_comment_model.h5')

with open('vietnamese_tfidf_vectorizer.pkl', 'rb') as f:
    vietnamese_vect = pickle.load(f)
vietnamese_model = load_model('vietnamese_tfidf_vectorizer.h5')


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
            return "Toxic Comment"
        else:
            return "Non-Toxic Comment"
    
    except Exception as e:
        return f"Error in detecting language or processing text: {e}"


new_text_english = "You’re an idiot person, and I hope someone hits you!"
# new_text_vietnamese = "Cút mẹ mày đi thằng ngu này"
new_text_vietnamese = "Vấn đề của Mp3 là chưa thích nghi với đội bóng mới và môi trường mới. Còn thích nghi được hay không thì phải qua một mùa giải mới có đánh giá chính xác. Ở League 1 thì PSG làm trùm nên cũng khó đánh giá đúng thực chất, vì khi ra C1 thì PSG cũng chỉ ở dạng trung bình khá. LaLiga nó ở một đẳng cấp khác xa so với League 1."
print(predict_toxicity(new_text_english))
print(predict_toxicity(new_text_vietnamese))
