import imaplib
import email
from email.policy import default
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

EMAIL = "complaintnlp@gmail.com"
PASSWORD = "qtmw towt jpjr zmfm"
IMAP_SERVER = "imap.gmail.com"

mailbox = imaplib.IMAP4_SSL(IMAP_SERVER)
mailbox.login(EMAIL, PASSWORD)
mailbox.select("INBOX")

status, messages = mailbox.search(None, '(SUBJECT "complaint")')

complaints = []
for msg_id in messages[0].split():
    status, msg_data = mailbox.fetch(msg_id, '(RFC822)')
    for response_part in msg_data:
        if isinstance(response_part, tuple):
            raw_msg = response_part[1]
            email_msg = email.message_from_bytes(raw_msg, policy=default)

            if email_msg.is_multipart():
                for part in email_msg.walk():
                    if part.get_content_type() == "text/plain":
                        complaint_text = part.get_payload(decode=True).decode().strip()
                        complaints.append(complaint_text)
            else:
                complaints.append(email_msg.get_payload(decode=True).decode().strip())

mailbox.logout()

ner_data = []
for complaint in complaints:
    doc = nlp(complaint)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    ner_data.append({"Complaint": complaint, "Entities": entities})

df = pd.DataFrame(ner_data)
df.to_excel("customer_complaints_ner.xlsx", index=False)

print("Complaints with NER saved to customer_complaints_ner.xlsx")
