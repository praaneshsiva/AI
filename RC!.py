import pandas as pd
import re
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from fastapi import FastAPI
from pydantic import BaseModel

file_path = "UpdatedResumeDataSet.csv"
df = pd.read_csv(file_path).sample(n=15, random_state=42)  # Using only 15 samples for quick testing


def clean_text(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["Cleaned_Resume"] = df["Resume"].apply(clean_text)
categories = df["Category"].unique()
category_to_id = {category: idx for idx, category in enumerate(categories)}
df["Label"] = df["Category"].map(category_to_id)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Cleaned_Resume"], df["Label"], test_size=0.2, random_state=42
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)


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

train_dataset = ResumeDataset(train_encodings, train_labels.tolist())
val_dataset = ResumeDataset(val_encodings, val_labels.tolist())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(categories))
model.to(device)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

model.save_pretrained("resume_classifier_model")
tokenizer.save_pretrained("resume_classifier_model")

app = FastAPI()


# Input Schema for API
class ResumeInput(BaseModel):
    resume_text: str


@app.post("/predict")
def predict_resume(input_data: ResumeInput):
    inputs = tokenizer(input_data.resume_text, truncation=True, padding=True, max_length=512, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to the same device as the model

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return {"category": categories[predicted_class]}
