# Safe Prompt Detection

Determine if prompt is malicious or safe

📊 What You'll See:

- Data split information: Shows train/val/test sizes
- Epoch progress: Detailed loss and accuracy for each epoch
- Validation monitoring: Real-time overfitting detection
- Best model tracking: Automatic saving of best performing model
- Early stopping: Training stops when validation loss plateaus
- Final test evaluation: Unbiased performance on held-out test set

## Binary Classification ("Safe" or "Malicious")

**Train**
```bash
python classifiers/binary.py --mode train \
    --epochs 10 \
    --batch_size 32 \
    --patience 5 \
    --test_size 0.15 \
    --val_size 0.15
```

**Evaluate**
```bash
# Evaluate on test set
python classifiers/binary.py --mode eval
```

**Predict**
```bash
# Make predictions
python classifiers/binary.py --mode predict --text "your text here"
```

## Multi-class Classification (Dangerous Content, Harassment, Sexually Explicit Information, Hate Speech)

**Train**
```bash
python safety_model.py --mode train --epochs 15 --batch_size 64 --patience 7 --use_class_weights --learning_rate 1e-5
```

**Evaluate**
```bash
# Evaluate on test set
python safety_model.py --mode eval
```

**Predict**
```bash
# Make predictions
python safety_model.py --mode predict --text "your text here"
```

## Training

```bash
==================================================
FINAL TEST EVALUATION
==================================================
Evaluating: 100%|██████████████████████████████████████████████| 16/16 [00:01<00:00,  9.81it/s]
                               precision    recall  f1-score   support

sexually explicit information       1.00      1.00      1.00        99
                   harassment       1.00      1.00      1.00        97
                  hate speech       1.00      1.00      1.00       108
            dangerous content       1.00      1.00      1.00       146
                         safe       1.00      1.00      1.00       550

                     accuracy                           1.00      1000
                    macro avg       1.00      1.00      1.00      1000
                 weighted avg       1.00      1.00      1.00      1000

Final Test Loss: 0.0012, Test Accuracy: 1.0000
```