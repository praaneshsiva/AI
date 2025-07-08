import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy

nlp = spacy.load("en_core_web_sm")
text = "I am Praanesh"
tokens = word_tokenize(text)
print(tokens)
stop_words = set(stopwords.words("english"))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print( filtered_tokens)
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
print(stemmed_words)
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print(lemmatized_words)
doc = nlp(text)
for token in doc:
    print(f"{token.text} --> {token.pos_}")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")
