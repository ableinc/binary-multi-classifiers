import argparse
import os
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from pandas import Series
from classifiers.multilabel import TextDataset, load_model, PromptClassifier, train, save_model, evaluate, predict

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
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    # Example data (replace with real data loading)
    df = pd.read_csv(os.path.join("./datasets", "qualifire-safety-benchmark.csv"))
    texts = []
    labels = []
    for _, row in df.iterrows():
        texts.append(row['text'])
        labels.append(LABEL2ID[get_label(row)])

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = TextDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(args.save_dir, MODEL_NAME)
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = load_model(args.save_dir, device, LABELS)
    else:
        print("Initializing new model.")
        model = PromptClassifier(labels=LABELS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    if args.mode == 'train':
        for epoch in range(args.epochs):
            avg_loss = train(model, dataloader, optimizer, criterion, device)
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
        save_model(model, args.save_dir)
        print(f"Model saved to {args.save_dir}")
    elif args.mode == 'eval':
        evaluate(model, dataloader, device, LABELS)
    elif args.mode == 'predict':
        if len(args.text) == 0:
            raise ValueError("text cannot be empty")
        answer = predict(model, tokenizer, args.text, device, ID2LABEL)
        print(answer)