import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from matplotlib import pyplot as plt
import pickle
from tensorflow.keras.models import load_model
%matplotlib inline
import seaborn as sns
import re
from wordcloud import WordCloud

!pip install pyvi
from pyvi import ViTokenizer
!pip install underthesea


# Load datasets
df = pd.read_csv('/kaggle/input/toxic-comment-dataset/translated_toxic_comment.csv')

# Process df
df['Toxic'] = df.iloc[:, 3:].any(axis=1).astype(np.int8)
df = df[['translated_comment_text', 'Toxic']].rename(columns={'translated_comment_text': 'Text'})
df = df.drop_duplicates(subset=['Text']).reset_index(drop=True)

from underthesea import word_tokenize
def clean_text_vietnamese(text):
    text = text.lower()
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = word_tokenize(text) 
    text = ' '.join(text)
    return text

df['Text'] = df['Text'].map(lambda com: clean_text_vietnamese(com))

vect_vietnamese = TfidfVectorizer(max_features=2000)
X_vietnamese = vect_vietnamese.fit_transform(df['Text'])
y_vietnamese = df['Toxic']

# Apply SMOTE to handle class imbalance
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_vietnamese, y_vietnamese)

# Split the resampled data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

train_data = tf.data.Dataset.from_tensor_slices((x_train.toarray(), y_train)).batch(32).shuffle(1000)
test_data = tf.data.Dataset.from_tensor_slices((x_test.toarray(), y_test)).batch(32)

model = Sequential([
    Dense(128, activation='relu', input_dim=X_resampled.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, epochs=5, validation_data=test_data)

# Plot training & validation accuracy values
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

loss, accuracy = model.evaluate(x_test.toarray(), y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

y_pred = (model.predict(x_test.toarray()) >= 0.5).astype(int)
print(classification_report(y_test, y_pred))


# Save the TF-IDF vectorizer and model
with open('vietnamese_tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vect_vietnamese, f)

model.save('vietnamese_toxic_comment_model.h5')

def predict_toxicity(text):
    # Load vectorizer and model
    with open('vietnamese_tfidf_vectorizer.pkl', 'rb') as f:
        loaded_vect = pickle.load(f)
    loaded_model = load_model('vietnamese_toxic_comment_model.h5')
    
    # Clean and transform text
    cleaned_text = clean_text_vietnamese(text)
    text_vector = loaded_vect.transform([cleaned_text])
    
    # Make prediction
    prediction = loaded_model.predict(text_vector.toarray())[0][0]
    return "Toxic Comment" if prediction >= 0.5 else "Non-Toxic Comment"

# Test prediction
test_text = "Thôi ông im mẹ mồm đi"
print(predict_toxicity(test_text))