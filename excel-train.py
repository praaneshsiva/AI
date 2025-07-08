import pandas as pd
df = pd.read_excel('customer_complaints.xlsx')
complaint_column = 'complaints'
def assign_category(complaint):
    complaint = str(complaint).lower()
    if any(word in complaint for word in ["billing", "charge", "payment"]):
        return "Billing Issue"
    elif any(word in complaint for word in ["service", "support", "customer care"]):
        return "Service Issue"
    elif any(word in complaint for word in ["delivery", "shipment", "delay"]):
        return "Delivery Issue"
    elif any(word in complaint for word in ["fraud", "scam", "unauthorized"]):
        return "Fraud Issue"
    else:
        return ""
df['category'] = df[complaint_column].apply(assign_category)

df.to_excel('training_dataset.xlsx', index=False)

print("Training dataset created successfully! Check 'training_dataset.xlsx' to manually label the remaining complaints.")
