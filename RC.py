import os
import pypandoc
import pdfplumber
import docx
import win32com.client
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import nltk
import numpy as np
import re

stop_words = set(stopwords.words('english'))

priority_words = {"python": 2.0, "machine learning": 2.5, "deep learning": 2.5, "design": 1.8, "management": 1.5,
                  "project": 1.2}

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text.strip()

def extract_text_from_docx(doc_path):
    doc = docx.Document(doc_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def extract_text_from_doc(doc_path):
    word = win32com.client.Dispatch("Word.Application")
    doc = word.Documents.Open(doc_path)
    text = doc.Range().Text
    doc.Close()
    word.Quit()
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words or len(word) > 2]
    return ' '.join(words)

class CustomTfidfVectorizer(TfidfVectorizer):
    def transform(self, raw_documents):
        X = super().transform(raw_documents)
        feature_names = self.get_feature_names_out()
        for word, weight in priority_words.items():
            if word in feature_names:
                index = np.where(feature_names == word)[0][0]
                X[:, index] *= weight
        return X

def load_dataset(selected_categories):
    data, labels, category_map = [], [], {}
    for idx, category in enumerate(selected_categories):
        category_map[idx] = category
        folder_path = os.path.join("resume_data", category)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".pdf"):
                    raw_text = extract_text_from_pdf(os.path.join(folder_path, file))
                    cleaned_text = preprocess_text(raw_text)
                    if cleaned_text:
                        data.append(cleaned_text)
                        labels.append(idx)
                elif file.endswith(".docx"):
                    raw_text = extract_text_from_docx(os.path.join(folder_path, file))
                    cleaned_text = preprocess_text(raw_text)
                    if cleaned_text:
                        data.append(cleaned_text)
                        labels.append(idx)
                elif file.endswith(".doc"):
                    raw_text = extract_text_from_doc(os.path.join(folder_path, file))
                    cleaned_text = preprocess_text(raw_text)
                    if cleaned_text:
                        data.append(cleaned_text)
                        labels.append(idx)
    return data, labels, category_map

class ResumeClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Resume Classifier")
        self.root.geometry("500x450")

        tk.Label(root, text="Select Job Categories", font=("Arial", 14)).pack(pady=5)

        self.available_categories = [folder for folder in os.listdir("resume_data") if
                                     os.path.isdir(os.path.join("resume_data", folder))]
        self.selected_categories = []

        self.job_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, height=6)
        for category in self.available_categories:
            self.job_listbox.insert(tk.END, category)
        self.job_listbox.pack(pady=5)

        tk.Button(root, text="Select or Add New Job", command=self.select_job, font=("Arial", 12)).pack(pady=10)
        self.result_label = tk.Label(root, text="", font=("Arial", 12), wraplength=400)
        self.result_label.pack(pady=10)

    def select_job(self):
        selected_indices = self.job_listbox.curselection()
        self.selected_categories = [self.available_categories[i] for i in selected_indices]

        if not self.selected_categories:
            response = messagebox.askyesno("New Job Category", "No job selected! Would you like to add a new category?")
            if response:
                self.open_new_category_window()
            return

        self.upload_file()

    def open_new_category_window(self):
        self.new_window = tk.Toplevel(self.root)
        self.new_window.title("Add New Job Category")
        self.new_window.geometry("400x300")

        tk.Label(self.new_window, text="Enter New Job Name:", font=("Arial", 12)).pack(pady=5)
        self.job_entry = tk.Entry(self.new_window, font=("Arial", 12))
        self.job_entry.pack(pady=5)

        tk.Label(self.new_window, text="Upload at least 5 resumes:", font=("Arial", 12)).pack(pady=5)
        tk.Button(self.new_window, text="Upload Resumes", command=self.upload_new_resumes, font=("Arial", 12)).pack(pady=5)

    def upload_new_resumes(self):
        job_name = self.job_entry.get().strip()
        if not job_name:
            messagebox.showerror("Error", "Job name cannot be empty!")
            return

        job_folder = os.path.join("resume_data", job_name)
        os.makedirs(job_folder, exist_ok=True)

        file_paths = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf"), ("Word Files", "*.docx;*.doc")])
        if len(file_paths) < 5:
            messagebox.showerror("Error", "You must upload at least 5 resumes for training!")
            return

        for file in file_paths:
            file_name = os.path.basename(file)
            os.rename(file, os.path.join(job_folder, file_name))

        messagebox.showinfo("Success", f"New job category '{job_name}' added successfully with {len(file_paths)} resumes!")
        self.new_window.destroy()
        self.available_categories.append(job_name)
        self.job_listbox.insert(tk.END, job_name)

    def upload_file(self):
        data, labels, category_map = load_dataset(self.selected_categories)

        if not data:
            messagebox.showerror("Error", "No valid resumes found in the selected categories!")
            return

        vectorizer = CustomTfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(data)
        y = np.array(labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "SVM": SVC(kernel='linear'),
            "Naïve Bayes": MultinomialNB(),
            "Random Forest": RandomForestClassifier(n_estimators=100)
        }

        for model in models.values():
            model.fit(X_train, y_train)

        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf"), ("Word Files", "*.docx;*.doc")])
        if not file_path:
            return

        if file_path.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file_path.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        else:
            text = extract_text_from_doc(file_path)

        processed_text = preprocess_text(text)
        X_input = vectorizer.transform([processed_text])

        predictions = {name: model.predict(X_input)[0] for name, model in models.items()}
        result_text = "\n".join([f"{name}: {category_map[pred]}" for name, pred in predictions.items()])
        self.result_label.config(text=result_text)
        messagebox.showinfo("Classification Result", result_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = ResumeClassifierApp(root)
    root.mainloop()
