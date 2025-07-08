import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from textblob import TextBlob

data = pd.read_csv("Customer_Complaints1.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text
def preprocess_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

data['Processed_Complaint'] = data['Complaint'].apply(preprocess_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Processed_Complaint'])
y = data['Category']

svm_model = SVC(kernel='linear')
svm_model.fit(X, y)

test_data = pd.read_excel("customer_complaints_mail.xlsx")
test_data['Processed_Complaint'] = test_data['Complaint'].apply(preprocess_text)

X_test = vectorizer.transform(test_data['Processed_Complaint'])


category_predictions = svm_model.predict(X_test)

def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'
test_data['Category'] = category_predictions
test_data['Sentiment'] = test_data['Complaint'].apply(get_sentiment)
test_data[['Complaint', 'Category', 'Sentiment']].to_csv("Predicted_Results1.csv", index=False)
print(test_data[['Complaint', 'Category', 'Sentiment']])
