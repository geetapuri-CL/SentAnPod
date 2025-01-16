import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

# STEP 1: Load the dataset
df = pd.read_csv("/data/CombinedFinancialEmotionDataset.csv")

# Check the dataset
#print(df.head())       # Display the first few rows
#print(df.info())       # Check for missing data
#print(df['emotion'].value_counts())  # Check the distribution of emotions
#df = df.drop_duplicates()

# STEP 2: Split the dataset
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['emotion'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['emotion'])

# Save the splits for future use
# uncomment following 6 lines if any change in data
"""
train_df.to_csv("/data/train.csv", index=False)
val_df.to_csv("/data/val.csv", index=False)
test_df.to_csv("/data/test.csv", index=False)

print(f"Training set: {len(train_df)} rows")
print(f"Validation set: {len(val_df)} rows")
print(f"Test set: {len(test_df)} rows")
"""


# STEP 3: Load a pre-trained tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the text
def tokenize_data(data):
    return tokenizer(list(data['sentence']), padding=True, truncation=True, max_length=128, return_tensors="pt")

train_encodings = tokenize_data(train_df)
val_encodings = tokenize_data(val_df)
test_encodings = tokenize_data(test_df)


# STEP4: Create dataset object
class FinancialDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx])
        }

# Map emotions to numeric labels
emotion_to_label = {"joy": 0, "neutral": 1, "fear": 2}
train_labels = [emotion_to_label[e] for e in train_df['emotion']]
val_labels = [emotion_to_label[e] for e in val_df['emotion']]
test_labels = [emotion_to_label[e] for e in test_df['emotion']]

# Create datasets
train_dataset = FinancialDataset(train_encodings, train_labels)
val_dataset = FinancialDataset(val_encodings, val_labels)
test_dataset = FinancialDataset(test_encodings, test_labels)


# STEP 5: Load a pre-trained model for classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(emotion_to_label))


# STEP 6: Fine Tune: Training arguments
training_args = TrainingArguments(
    output_dir="/data/results",        # Save checkpoints and model
    evaluation_strategy="epoch",  # Evaluate after each epoch
    learning_rate=2e-5,           # Learning rate
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="/data/logs",
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# STEP 7: Evaluate on test dataset
results = trainer.evaluate(test_dataset)
print(results)

# STEP 8: Save the fine tuned model
model.save_pretrained("/data/model/financial_emotion_model")
tokenizer.save_pretrained("/data/model/financial_emotion_model")




