from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from classifiers.multilabel import load_model, predict
from classifiers.utils import setup_device
from transformers import AutoTokenizer
import os
import uvicorn

app = FastAPI(title="Safety API")

model_path = os.path.join("./", "safety_model.pt")
if not os.path.exists(model_path):
    raise AttributeError('safety_model.pt file not found')

# Setup device and multi-GPU
device, use_multi_gpu = setup_device()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
LABELS = ["sexually explicit information", "harassment", "hate speech", "dangerous content", "safe"]
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
# Load model
model = load_model(model_path, device, LABELS)

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    message: str
    safe: bool


def process(text: str):
    return predict(model, tokenizer, text, device, ID2LABEL)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks
):
    try:
        answer = process(request.text)
        return AnalyzeResponse(
            message=answer,
            safe=answer == "safe"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="something went wrong"
        )


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=4000)