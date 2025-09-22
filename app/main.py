import os, yaml
from fastapi import FastAPI, UploadFile, File 
from .state import state 

app = FastAPI(title="YOLOv5-mini Toy")

@app.post("/train")
async def train():
  if state.running:
    return {"status": "already running"}
  cfg = yaml.safeload(open('config.yaml'))
  os.makedirs('runs/exp1', exist_ok=True)
  state.running = True 
  state.epoch = 0
  state.total_epochs = cfg['epochs']
  state.history = []

  return {"status":"done", "metrics":state.history}