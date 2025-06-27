import os
import argparse
from transformers import AutoTokenizer, DistilBertModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from .utils import setup_device, split_data, EarlyStopping, save_checkpoint, load_checkpoint, calculate_class_weights, print_training_info

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
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # binary class

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0]  # [CLS] token
        return self.classifier(cls_output)

# Training loop with progress tracking
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
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
        
        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

# Evaluation loop with progress tracking
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    print(classification_report(all_labels, all_preds, target_names=["safe", "malicious"]))
    return avg_loss, accuracy

# Prediction
def predict(model, tokenizer, text, device):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs)
        pred = torch.argmax(logits, dim=1).item()
        return bool(pred)

# Model save/load
def save_model(model, path, model_name):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, model_name))

def load_model(model_path, device):
    model = PromptClassifier()
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle DataParallel state dict (keys prefixed with 'module.')
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    return model

# Training function with validation and early stopping
def train_with_validation(model, train_dataloader, val_dataloader, optimizer, criterion, 
                         device, epochs, save_dir, model_name, patience=5):
    """Train model with validation and early stopping."""
    early_stopping = EarlyStopping(patience=patience)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, device)
        
        # Validation
        val_loss, val_acc = evaluate(model, val_dataloader, device)
        
        # Print training info
        print_training_info(epoch+1, epochs, train_loss, val_loss, train_acc, val_acc) # type: ignore
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, save_dir, model_name)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, save_dir, model_name)
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or evaluate the binary PromptClassifier.")
    parser.add_argument('--mode', choices=['train', 'eval', 'predict'], required=True)
    parser.add_argument('--text', type=str, default="")
    parser.add_argument('--save_dir', type=str, default="./binary")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of data for test set")
    parser.add_argument('--val_size', type=float, default=0.1, help="Proportion of remaining data for validation set")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--use_class_weights', action='store_true', help="Use class weights for imbalanced data")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    # Example data (replace with real data loading)
    texts = ["This is safe.", "This is malicious!"] * 100
    labels = [0, 1] * 100
    
    # Split data into train/validation/test sets
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(
        texts, labels, test_size=args.test_size, val_size=args.val_size
    )
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Setup device and multi-GPU
    device, use_multi_gpu = setup_device()
    
    model_path = os.path.join(args.save_dir, "model.pt")
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = load_model(model_path, device)
    else:
        print("Initializing new model.")
        model = PromptClassifier().to(device)
    
    # Use DataParallel for multi-GPU
    if use_multi_gpu:
        model = nn.DataParallel(model)
    
    # Setup optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    if args.use_class_weights:
        print("Using class weights for imbalanced data")
        class_weights = calculate_class_weights(train_labels, 2).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    if args.mode == 'train':
        print(f"Starting training with {args.epochs} epochs, batch size {args.batch_size}")
        print(f"Data split: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")
        
        # Train with validation and early stopping
        model = train_with_validation(
            model, train_dataloader, val_dataloader, optimizer, criterion,
            device, args.epochs, args.save_dir, "model.pt", args.patience
        )
        
        # Final evaluation on test set
        print("\n" + "="*50)
        print("FINAL TEST EVALUATION")
        print("="*50)
        test_loss, test_acc = evaluate(model, test_dataloader, device)
        print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        
    elif args.mode == 'eval':
        print("Evaluating on test set...")
        test_loss, test_acc = evaluate(model, test_dataloader, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        
    elif args.mode == 'predict':
        if len(args.text) == 0:
            raise ValueError("text cannot be empty")
        answer = predict(model, tokenizer, args.text, device)
        print(f"Bad prompt: {answer}")
