import os, yaml
from fastapi import FastAPI, UploadFile, File 
from .state import state 
from training.train_loop import Trainer 
from training.inference import Predictor

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

  trainer = Trainer(cfg)


  return {"status":"done", "metrics":state.history}

@app.get("/progress")
async def progress():
  return {
    "running": state.running,
    "epoch": state.epoch,
    "total_epochs": state.total_epochs,
    "val_acc": state.val_acc 
  }

@app.post("/predict")
async def predict(file:UploadFile = File(...)):
  path = f"/tmp/{file.filename}" # file.filename = data/images/val/00000.jpg
  with open(path, 'wb') as f:
    f.write(await file.read())
  pred = Predictor()(path)
  return {"pred": pred}
  