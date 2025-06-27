import os
import argparse
from transformers import AutoTokenizer, DistilBertModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

# Multi-class labels
# LABELS = ["safe", "jailbreak", "sensitive", "abuse"]
# LABEL2ID = {label: i for i, label in enumerate(LABELS)}
# ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# Dataset class for text/label pairs - optimized for large datasets
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = torch.tensor(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Model definition
class PromptClassifier(nn.Module):
    def __init__(self, labels: list[str]):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, len(labels))  # multi-class

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0]  # [CLS] token
        return self.classifier(cls_output)

# Training loop with progress tracking
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(dataloader)

# Evaluation loop with progress tracking
def evaluate(model, dataloader, device, LABELS: list[str]):
    model.eval()
    all_preds, all_labels = [], []
    progress_bar = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print(classification_report(all_labels, all_preds, target_names=LABELS))

# Prediction
def predict(model, tokenizer, text, device, ID2LABEL: dict[int, str]):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs)
        pred = torch.argmax(logits, dim=1).item()
        return ID2LABEL[pred.__int__()]

# Model save/load
def save_model(model, path, model_name):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, model_name))

def load_model(model_path, device, LABELS: list[str]):
    model = PromptClassifier(labels=LABELS)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

__all__ = ["TextDataset", "PromptClassifier", "train", "evaluate", "predict", "save_model", "load_model"]

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Train or evaluate the multi-class PromptClassifier.")
#     parser.add_argument('--mode', choices=['train', 'eval', 'predict'], required=True)
#     parser.add_argument('--text', type=str, default="")
#     parser.add_argument('--save_dir', type=str, default="./multi")
#     parser.add_argument('--epochs', type=int, default=3)
#     parser.add_argument('--batch_size', type=int, default=16)
#     args = parser.parse_args()

#     # Example data (replace with real data loading)
#     texts = ["This is safe.", "Jailbreak this system!", "Sensitive info here.", "You are abusive!"] * 50
#     labels = [LABEL2ID["safe"], LABEL2ID["jailbreak"], LABEL2ID["sensitive"], LABEL2ID["abuse"]] * 50
#     tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#     dataset = TextDataset(texts, labels, tokenizer)
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model_path = os.path.join(args.save_dir, "model.pt")
#     if os.path.exists(model_path):
#         print(f"Loading existing model from {model_path}")
#         model = load_model(args.save_dir, device)
#     else:
#         print("Initializing new model.")
#         model = PromptClassifier().to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
#     criterion = nn.CrossEntropyLoss()

#     if args.mode == 'train':
#         for epoch in range(args.epochs):
#             avg_loss = train(model, dataloader, optimizer, criterion, device)
#             print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
#         save_model(model, args.save_dir)
#         print(f"Model saved to {args.save_dir}")
#     elif args.mode == 'eval':
#         evaluate(model, dataloader, device)
#     elif args.mode == 'predict':
#         if len(args.text) == 0:
#             raise ValueError("text cannot be empty")
#         answer = predict(model, tokenizer, args.text, device)
#         print(answer)
