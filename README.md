# Safe Prompt Detection

Detect if the prompt being sent to the LLM is safe or not.

## Binary Classification ("Safe" or "Malicious")

**Train**
```bash
python model-binary.py --mode train --epochs 3 --batch_size 16
```

**Evaluate**
```bash
python model-binary.py --mode eval
```

**Predict**
```bash
python model-binary.py --mode predict --text "this is malice"
```

## Multi-class Classification ("Safe", "Jailbreak", "Sensitive", "Abuse")

**Train**
```bash
python model-multi-class.py --mode train --epochs 3 --batch_size 16
```

**Evaluate**
```bash
python model-multi-class.py --mode eval
```

**Predict**
```bash
python model-multi-class.py --mode predict --text "this is a jailbreak prompt"
```