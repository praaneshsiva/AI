import pandas as pd
import re
import numpy as np
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import json
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

stop_words = set(stopwords.words('english'))

priority_words = {
    "python": 2.0,
    "machine learning": 2.5,
    "deep learning": 2.5,
    "design": 1.8,
    "management": 1.5,
    "project": 1.2
}

class CustomTfidfVectorizer(TfidfVectorizer):
    def transform(self, raw_documents):
        X = super().transform(raw_documents)
        feature_names = self.get_feature_names_out()
        for word, weight in priority_words.items():
            if word in feature_names:
                index = np.where(feature_names == word)[0][0]
                X[:, index] *= weight
        return X

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

df = pd.read_csv('UpdatedResumeDataSet.csv')
df = df[['Resume', 'Category']].dropna()
df['Cleaned_Resume'] = df['Resume'].apply(preprocess_text)

labels = df['Category'].astype('category')
df['Label'] = labels.cat.codes
category_map = dict(enumerate(labels.cat.categories))

with open('category_map.json', 'w') as f:
    json.dump(category_map, f)

vectorizer = CustomTfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['Cleaned_Resume'])
y = df['Label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "SVM": SVC(kernel='linear'),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

model_accuracies = {}

for name, model in models.items():
    print(f"\n{name} Results:")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred) * 100
    model_accuracies[name] = acc

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=labels.cat.categories, zero_division=0)

    print(f"Accuracy: {acc:.2f}%")
    print("Classification Report:\n", report)

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels.cat.categories, yticklabels=labels.cat.categories)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(8, 4))
sns.barplot(
    x=list(model_accuracies.keys()),
    y=list(model_accuracies.values()),
    hue=list(model_accuracies.keys()),
    palette="viridis",
    legend=False
)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

