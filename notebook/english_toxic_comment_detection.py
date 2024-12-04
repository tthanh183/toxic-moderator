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

df1 = pd.read_csv('/kaggle/input/10022024/train.csv')
df2 = pd.read_csv('/kaggle/input/10022024/youtoxic_english_1000.csv')

df1['Toxic'] = df1.iloc[:, 2:].any(axis=1).astype(np.int8)
df2['Toxic'] = df2.iloc[:, 3:].any(axis=1).astype(np.int8)
df1_processed = df1[['comment_text', 'Toxic']].rename(columns={'comment_text': 'Text'})
df2_processed = df2[['Text', 'Toxic']]
df = pd.concat([df1_processed, df2_processed], ignore_index=True).drop_duplicates(subset=['Text']).reset_index(drop=True)

def clean_text(text):
    import re
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
    return text.strip()

# Clean text data
df['Text'] = df['Text'].map(clean_text)

vect = TfidfVectorizer(max_features=2000, stop_words='english')
X = vect.fit_transform(df['Text'])
y = df['Toxic']

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

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


with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vect, f)
model.save('toxic_comment_model.h5')


def predict_toxicity(text):
    # Load vectorizer and model
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        loaded_vect = pickle.load(f)
    loaded_model = load_model('toxic_comment_model.h5')
    
    # Clean and transform text
    cleaned_text = clean_text(text)
    text_vector = loaded_vect.transform([cleaned_text])
    
    # Make prediction
    prediction = loaded_model.predict(text_vector.toarray())[0][0]
    return "Toxic Comment" if prediction >= 0.5 else "Non-Toxic Comment"

# Test prediction
test_text = "oh shit, you're idiot. get out of my eye"
print(predict_toxicity(test_text))
