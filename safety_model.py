import argparse
import os
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from pandas import Series
from classifiers.multilabel import TextDataset, load_model, PromptClassifier, train_with_validation, save_model, evaluate, predict
from classifiers.utils import setup_device, split_data, calculate_class_weights
import gc

# Multi-class labels
LABELS = ["sexually explicit information", "harassment", "hate speech", "dangerous content", "safe"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
MODEL_NAME = "safety_model.pt"

def get_label(df: Series):
    if df["Sexually Explicit Information"] == 1:
        return "sexually explicit information"
    if df["Harassment"] == 1:
        return "harassment"
    if df["Hate Speech"] == 1:
        return "hate speech"
    if df["Dangerous Content"] == 1:
        return "dangerous content"
    return "safe"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train, evaluate or predict multi-label PromptClassifier.")
    parser.add_argument('--mode', choices=['train', 'eval', 'predict'], required=True)
    parser.add_argument('--text', type=str, default="")
    parser.add_argument('--save_dir', type=str, default="./")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of data for test set")
    parser.add_argument('--val_size', type=float, default=0.1, help="Proportion of remaining data for validation set")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--use_class_weights', action='store_true', help="Use class weights for imbalanced data")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    torch.cuda.empty_cache()
    gc.collect()

    # Setup device and multi-GPU
    device, use_multi_gpu = setup_device()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Predict using model
    if args.mode == 'predict':
        if len(args.text) == 0:
            raise ValueError("text cannot be empty")
        
        model_path = os.path.join(args.save_dir, MODEL_NAME)
        if not os.path.exists(model_path):
            raise ValueError(f"Model not found at {model_path}")
        
        model = load_model(model_path, device, LABELS)
        answer = predict(model, tokenizer, args.text, device, ID2LABEL)
        print(answer)
        exit(0)

    # Load data
    print("Loading dataset...")
    df = pd.read_csv(os.path.join("./datasets", "qualifire-safety-benchmark.csv"))
    texts = []
    labels = []
    for _, row in df.iterrows():
        texts.append(row['text'])
        labels.append(LABEL2ID[get_label(row)])

    print(f"Dataset loaded: {len(texts)} samples")
    
    # Split data into train/validation/test sets
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(
        texts, labels, test_size=args.test_size, val_size=args.val_size
    )
    
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

    # Load existing model or create new one
    model_path = os.path.join(args.save_dir, MODEL_NAME)
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = load_model(model_path, device, LABELS)
    else:
        print("Initializing new model.")
        model = PromptClassifier(labels=LABELS).to(device)
    
    # Use DataParallel for multi-GPU
    if use_multi_gpu:
        model = nn.DataParallel(model)
    
    # Setup optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    if args.use_class_weights:
        print("Using class weights for imbalanced data")
        class_weights = calculate_class_weights(train_labels, len(LABELS)).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    if args.mode == 'train':
        print(f"Starting training with {args.epochs} epochs, batch size {args.batch_size}")
        print(f"Data split: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")
        
        # Train with validation and early stopping
        model = train_with_validation(
            model, train_dataloader, val_dataloader, optimizer, criterion,
            device, args.epochs, args.save_dir, MODEL_NAME, LABELS, args.patience
        )
        
        # Final evaluation on test set
        print("\n" + "="*50)
        print("FINAL TEST EVALUATION")
        print("="*50)
        test_loss, test_acc = evaluate(model, test_dataloader, device, LABELS)
        print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        
    elif args.mode == 'eval':
        print("Evaluating on test set...")
        test_loss, test_acc = evaluate(model, test_dataloader, device, LABELS)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")