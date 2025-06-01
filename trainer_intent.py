import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import joblib

# ✅ 1. Load dataset with explicit encoding
csv_path = "synthetic_email_intent_dataset_v2.csv"
df = pd.read_csv(csv_path, encoding='utf-8')  # Added explicit encoding

# ✅ 2. Handle missing values
df = df.dropna(subset=['body', 'intent'])  # Added data cleaning

# ✅ 3. Encode labels
label_encoder = LabelEncoder()
df['encoded_intent'] = label_encoder.fit_transform(df['intent'])

# ✅ 4. Stratified split for class balance
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['body'].tolist(),
    df['encoded_intent'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['intent']  # Added stratification
)

# ✅ 5. Load tokenizer with error handling
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
except Exception as e:
    print(f"Error loading tokenizer: {str(e)}")

# ✅ 6. Fixed dataset class
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])  # Ensure string type
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        # Removed unnecessary flatten() - keeps batch dimension
        return {
            'input_ids': encoding['input_ids'][0],  # [0] to remove batch dimension
            'attention_mask': encoding['attention_mask'][0],
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ✅ 7. Create datasets with validation
train_dataset = EmailDataset(train_texts, train_labels, tokenizer)
test_dataset = EmailDataset(test_texts, test_labels, tokenizer)

# ✅ 8. Model loading with mismatch handling
try:
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(label_encoder.classes_),
        ignore_mismatched_sizes=True  # Critical fix for label mismatches
    )
except Exception as e:
    print(f"Error loading model: {str(e)}")

# ✅ 9. Enhanced training arguments
# ✅ 8. Updated training arguments
training_args = TrainingArguments(
    output_dir='./email_intent_bert',
    eval_strategy="epoch",  # Changed from evaluation_strategy to eval_strategy
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./email_intent_bert/logs',
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    load_best_model_at_end=True
)


# ✅ 10. Trainer with compute_metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# ✅ 11. Training with exception handling
try:
    trainer.train()
except RuntimeError as e:
    print(f"Training error: {str(e)}")
    if "CUDA out of memory" in str(e):
        print("Reduce batch size or use smaller model")

# ✅ 12. Safe saving
model.save_pretrained('./email_intent_bert')
tokenizer.save_pretrained('./email_intent_bert')
joblib.dump(label_encoder.classes_, './email_intent_bert/label_classes.pkl')
