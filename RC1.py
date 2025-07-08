import os
import pandas as pd
import docx
import pdfplumber
import win32com.client
import re
import nltk

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return '\n'.join([para.text for para in doc.paragraphs])

def extract_text_from_doc(path):
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = False
    try:
        doc = word.Documents.Open(os.path.abspath(path))
        text = doc.Range().Text
        doc.Close(False)
        return text.strip()
    except Exception as e:
        print(f"Error opening .doc file: {path} -> {e}")
        return ""
    finally:
        word.Quit()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words or len(w) > 2]
    return ' '.join(words)

def process_resume_folder(base_dir):
    data = []
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if os.path.isdir(category_path):
            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)
                text = ""
                if file.endswith(".pdf"):
                    text = extract_text_from_pdf(file_path)
                elif file.endswith(".docx"):
                    text = extract_text_from_docx(file_path)
                elif file.endswith(".doc"):
                    try:
                        text = extract_text_from_doc(file_path)
                    except Exception as e:
                        print(f"Failed to extract from {file}: {e}")
                        continue
                if text:
                    cleaned = preprocess_text(text)
                    data.append({"Resume": cleaned, "Category": category})
    return pd.DataFrame(data)

df = process_resume_folder("resume_data")
df.to_excel("ResumeDataset.xlsx", index=False)
print("Saved to ResumeDataset.xlsx")
