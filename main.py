from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration 
import re, json
import os
from peft import PeftModel

app = FastAPI(title="Fitness API")

# ---- Load your model once ----
MODEL_NAME = "./t5_fitness"  # replace with your model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained("./t5_fitness/checkpoint-2500")
model = PeftModel.from_pretrained(base_model, "./t5_fitness/checkpoint-2500").to(device)

# ---- API Key from environment variable ----
API_KEY = os.getenv("FITNESS_API_KEY", "haorudghoeurhgosuehr")  # fallback for testing

def verify_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

# ---- Request schema ----
class UserRequest(BaseModel):
    age: int
    height: float    # cm
    weight: float    # kg
    bmi: float 
    activity_level: str   # sedentary, moderate, active
    fitness_goal: str     # fat loss, muscle gain

# ---- JSON fixer ----
def fix_to_json(text: str):
    text = text.strip()
    if text.startswith("["):
        text = text[1:]
    if text.endswith("]"):
        text = text[:-1]
    text = re.sub(r'"day(\d+)":', r'"day\1": {', text)
    text = text.replace('},  "day', '},"day')
    if not text.startswith("{"):
        text = "{" + text
    if not text.endswith("}"):
        text = text + "}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_output": text}

# ---- API endpoint ----
@app.post("/get-plan")
def get_plan(user: UserRequest, valid: bool = Depends(verify_key)):
    # Build prompt
    prompt = f"""
    age: {user.age}
    weight: {user.weight}
    height: {user.height}
    bmi: {user.bmi}
    fitness_level: {user.activity_level}
    fitness_goal: {user.fitness_goal}
    """

    # Run model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=1500,
        do_sample=True,
        temperature=0.1,
        top_p=0.1
    )
    plan_text = tokenizer.decode(out[0], skip_special_tokens=True)
    plan_json = fix_to_json(plan_text)

    return {"plan": plan_json}

@app.get("/ping")
def ping():
    return {"status": "ok"}


# ---- Run with uvicorn ----
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
print(API_KEY)  