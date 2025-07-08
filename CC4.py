# Install required libraries
# !pip install transformers -q

import pandas as pd
import numpy as np
import re
import string
import nltk
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, TFBertForSequenceClassification

# Download NLTK data
nltk.download('punkt')

# === Step 1: Load and clean the data ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

data = pd.read_csv("Customer_Complaints1.csv")
data = data.dropna(subset=['Complaint', 'Category'])
data['Complaint'] = data['Complaint'].apply(clean_text)

# === Step 2: Encode the labels ===
le = LabelEncoder()
data['label'] = le.fit_transform(data['Category'])
num_classes = len(le.classes_)

# === Step 3: Split the dataset ===
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['Complaint'].tolist(), data['label'].tolist(), test_size=0.2, random_state=42)

# === Step 4: Tokenize the text using BERT tokenizer ===
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# === Step 5: Convert to TensorFlow datasets ===
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
)).batch(16)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_labels
)).batch(16)

# === Step 6: Load BERT model ===
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

# === Step 7: Compile the model ===
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# === Step 8: Train the model ===
history = model.fit(train_dataset, validation_data=test_dataset, epochs=3)

# === Step 9: Evaluate and print results ===
preds = model.predict(test_dataset).logits
y_pred = np.argmax(preds, axis=1)

print(classification_report(test_labels, y_pred, target_names=le.classes_))
