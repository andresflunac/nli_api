from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import numpy as np
import random as rd

def predict(premise,hypothesis):
    outputs = ['contradictory', 'neutral', 'entailed']
    # Encode inputs 
    x = tokenizer.encode(premise, hypothesis, return_tensors='pt')
    # Make Predictions
    logits = nli_model(x)[0]
    # Apply Softmax
    probs = softmax(logits.detach().numpy(), axis=1)[0]
    # Apply argmax
    prediction = np.argmax(np.array(probs))
    score = round(np.array(probs)[prediction],4)
    return outputs[prediction],score

app = FastAPI()
#model_name = 'facebook/bart-large-mnli'
model_name = 'joeddav/xlm-roberta-large-xnli'
nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/single")
def single_inference(stat_1:str, stat_2:str):
    prediction, score = predict(stat_1,stat_2)
    return {
        "inputs":
        {
            "statement_1": stat_1,
            "statement_2": stat_2
        },
        "outputs":
        {
            "inference": prediction,
            "confidence": str(score)
        }
    }

if __name__ == '__main__':

    # Development
    uvicorn.run('nli_api:app',port=8000, reload = True)

    # Deployment
    #uvicorn.run('nli_api:app', host = '0.0.0.0', port = 80, reload = True)