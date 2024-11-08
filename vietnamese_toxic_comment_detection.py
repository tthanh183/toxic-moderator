import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
import pickle

!pip install pyvi
from pyvi import ViTokenizer
!pip install underthesea

# Load datasets
df = pd.read_csv('/kaggle/input/toxic-comment-dataset/translated_toxic_comment.csv')


# Process df
df['Toxic'] = df.iloc[:, 3:].any(axis=1)
df = df[['translated_comment_text', 'Toxic']].rename(columns={'translated_comment_text': 'Text'})
df

# Check for missing values and drop duplicates
df.drop_duplicates(subset=['Text'], keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)


# Convert 'Toxic' to binary (True: 1, False: 0)
df['Toxic'] = df['Toxic'].astype(int)


from underthesea import word_tokenize
def clean_text_vietnamese(text):
    text = text.lower()
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = word_tokenize(text) 
    text = ' '.join(text)
    return text


df['Text'] = df['Text'].map(lambda com: clean_text_vietnamese(com))


vect_vietnamese = TfidfVectorizer(max_features=5000)
X_vietnamese = vect_vietnamese.fit_transform(df['Text'])
y_vietnamese = df['Toxic']


# Apply SMOTE to handle class imbalance
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_vietnamese, y_vietnamese)


# Split the resampled data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# Build the neural network model
model = Sequential([
    Dense(64, activation='relu', input_dim=X_resampled.shape[1]),  # Specify input dimension
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


# Train the model
history = model.fit(x_train.toarray(), y_train, epochs=5, batch_size=32, validation_split=0.2)

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


# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test.toarray(), y_test)
print(f'Test Accuracy: {accuracy}')


# Save the TF-IDF vectorizer and model
with open('vietnamese_tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vect_vietnamese, f)

model.save('vietnamese_tfidf_vectorizer.h5')


import pickle
from tensorflow.keras.models import load_model

# Load the saved TF-IDF vectorizer from output folder
with open('/kaggle/working/vietnamese_tfidf_vectorizer.pkl', 'rb') as f:
    vect = pickle.load(f)

# Load the saved Keras model from output folder
model = load_model('/kaggle/working/vietnamese_tfidf_vectorizer.h5')
def clean_text_vietnamese(text):
    text = text.lower()
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = word_tokenize(text) 
    text = ' '.join(text)
    return text

# Văn bản cần dự đoán
new_text = "Bố dí b thèm quan tâm đến mày"

# Làm sạch văn bản
cleaned_text = clean_text_vietnamese(new_text)

# Chuyển văn bản thành vector TF-IDF
text_vector = vect.transform([cleaned_text])
# Sử dụng mô hình để dự đoán
prediction = model.predict(text_vector.toarray())

# Hiển thị kết quả dự đoán
if prediction >= 0.5:
    print("Toxic Comment")
else:
    print("Non-Toxic Comment")
