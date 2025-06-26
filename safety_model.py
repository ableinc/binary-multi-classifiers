import argparse
import os
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from pandas import Series
from classifiers.multilabel import TextDataset, load_model, PromptClassifier, train, save_model, evaluate, predict, setup_device

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
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)  # Increased batch size
    parser.add_argument('--num_workers', type=int, default=4)  # Add num_workers for faster data loading
    args = parser.parse_args()

    # Load data
    print("Loading dataset...")
    df = pd.read_csv(os.path.join("./datasets", "qualifire-safety-benchmark.csv"))
    texts = []
    labels = []
    for _, row in df.iterrows():
        texts.append(row['text'])
        labels.append(LABEL2ID[get_label(row)])

    print(f"Dataset loaded: {len(texts)} samples")

    # Setup device and multi-GPU
    device, use_multi_gpu = setup_device()
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = TextDataset(texts, labels, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model_path = os.path.join(args.save_dir, MODEL_NAME)
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = load_model(args.save_dir, device, LABELS, MODEL_NAME)
    else:
        print("Initializing new model.")
        model = PromptClassifier(labels=LABELS).to(device)
    
    # Use DataParallel for multi-GPU
    if use_multi_gpu:
        model = nn.DataParallel(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    if args.mode == 'train':
        print(f"Starting training with {args.epochs} epochs, batch size {args.batch_size}")
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            avg_loss = train(model, dataloader, optimizer, criterion, device)
            print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save the model (unwrap from DataParallel if needed)
        if use_multi_gpu:
            model_to_save = model.module
        else:
            model_to_save = model
        save_model(model_to_save, args.save_dir, MODEL_NAME)
        print(f"Model saved to {args.save_dir}")
    elif args.mode == 'eval':
        evaluate(model, dataloader, device, LABELS)
    elif args.mode == 'predict':
        if len(args.text) == 0:
            raise ValueError("text cannot be empty")
        answer = predict(model, tokenizer, args.text, device, ID2LABEL)
        print(answer)