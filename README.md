# Safe Prompt Detection

Detect if the prompt being sent to the LLM is safe or not.

## Binary Classification ("Safe" or "Malicious")

**Train**
```bash
python classifiers/binary.py --mode train --epochs 10 --batch_size 64 --num_workers 4
```

**Evaluate**
```bash
python classifiers/binary.py --mode eval
```

**Predict**
```bash
python classifiers/binary.py --mode predict --text "your text here"
```

## Multi-class Classification ("Safe", "Jailbreak", "Sensitive", "Abuse")

**Train**
```bash
python classifiers/multilabel.py --mode train --epochs 10 --batch_size 64 --num_workers 4
```

**Evaluate**
```bash
python classifiers/binary.py --mode eval
```

**Predict**
```bash
python classifiers/multilabel.py --mode predict --text "this is a jailbreak prompt"
```