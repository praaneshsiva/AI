import pandas as pd
import re
import torch
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

file_path = "UpdatedResumeDataSet.csv"
df = pd.read_csv(file_path)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["Cleaned_Resume"] = df["Resume"].apply(clean_text)

categories = df["Category"].unique()
category_to_id = {category: idx for idx, category in enumerate(categories)}
id_to_category = {idx: category for category, idx in category_to_id.items()}
df["Label"] = df["Category"].map(category_to_id)

with open("label_mapping.json", "w") as f:
    json.dump(category_to_id, f)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Cleaned_Resume"],
    df["Label"],
    test_size=0.2,
    random_state=42,
    stratify=df["Label"]
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(
    list(train_texts), truncation=True, padding=True, max_length=512
)
val_encodings = tokenizer(
    list(val_texts), truncation=True, padding=True, max_length=512
)

class ResumeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

limited_train_encodings = {k: v[:24] for k, v in train_encodings.items()}
limited_train_labels = train_labels.tolist()[:24]

train_dataset = ResumeDataset(limited_train_encodings, limited_train_labels)
val_dataset = ResumeDataset(val_encodings, val_labels.tolist())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(categories))
model.to(device)

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

training_args = TrainingArguments(
    output_dir="./results",
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=1,
    save_steps=10,
    eval_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
model.save_pretrained("resume_classifier_model")
tokenizer.save_pretrained("resume_classifier_model")
