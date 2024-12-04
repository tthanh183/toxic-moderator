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


# Load datasets
df1 = pd.read_csv('/kaggle/input/10022024/train.csv')
df2 = pd.read_csv('/kaggle/input/10022024/youtoxic_english_1000.csv')

# Process df1
df1['Toxic'] = df1.iloc[:, 2:].any(axis=1)
df1_processed = df1[['comment_text', 'Toxic']].rename(columns={'comment_text': 'Text'})

# Process df2
df2['Toxic'] = df2.iloc[:, 3:].any(axis=1)
df2_processed = df2[['Text', 'Toxic']]

# Combine df1_processed and df2_processed
df = pd.concat([df1_processed, df2_processed], ignore_index=True)

# Check for missing values and drop duplicates
df.drop_duplicates(subset=['Text'], keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)

# Convert 'Toxic' to binary (True: 1, False: 0)
df['Toxic'] = df['Toxic'].astype(int)


# Clean text function
def clean_text(text):
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
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

# Apply clean_text to 'Text' column
df['Text'] = df['Text'].map(lambda com: clean_text(com))


# Vectorize text data using TF-IDF
vect = TfidfVectorizer(max_features=5000, stop_words='english')
X = vect.fit_transform(df['Text'])
y = df['Toxic']


# Apply SMOTE to handle class imbalance
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

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
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vect, f)

model.save('toxic_comment_model.h5')

import pickle
from tensorflow.keras.models import load_model

# Load the saved TF-IDF vectorizer from output folder
with open('/kaggle/working/tfidf_vectorizer.pkl', 'rb') as f:
    vect = pickle.load(f)

# Load the saved Keras model from output folder
model = load_model('/kaggle/working/toxic_comment_model.h5')
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
    text = text.strip()
    return text

# Văn bản cần dự đoán
new_text = "You’re an idiot person, and I hope someone hits you!"

# Làm sạch văn bản
cleaned_text = clean_text(new_text)

# Chuyển văn bản thành vector TF-IDF
text_vector = vect.transform([cleaned_text])
# Sử dụng mô hình để dự đoán
prediction = model.predict(text_vector.toarray())

# Hiển thị kết quả dự đoán
if prediction >= 0.5:
    print("Toxic Comment")
else:
    print("Non-Toxic Comment")
