import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from torch.nn.utils.rnn import pad_sequence

# Load dataset
df = pd.read_excel('preprocessed_complaints.xlsx')

# Drop rows with NaN values in the 'narrative' column
df = df.dropna(subset=['preprocessed_text'])

# Split data into features (X) and target labels (y)
X = df['preprocessed_text'].values
y = df['product'].values

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input text and pad sequences
tokenized_texts = [tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True) for text in X]
padded_texts = pad_sequence([torch.tensor(text) for text in tokenized_texts], batch_first=True, padding_value=tokenizer.pad_token_id)

# Create attention masks
attention_masks = (padded_texts != tokenizer.pad_token_id).type(torch.FloatTensor)

# Perform inference using BERT model
model = BertModel.from_pretrained('bert-base-uncased')
with torch.no_grad():
    outputs = model(input_ids=padded_texts, attention_mask=attention_masks)

# Use BERT embeddings for classification (you can use pooled_output or last_hidden_state)
pooled_output = outputs[0][:, 0, :].numpy()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(pooled_output, y, test_size=0.2, random_state=42)

# Train Support Vector Machine (SVM) model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predict labels on test set
y_pred = svm_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
