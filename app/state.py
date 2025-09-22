from typing import Optional

class TrainState:
  def __init__(self):
    self.running = False 
    self.epoch = 0
    self.total_epochs = 0
    self.val_acc = 0.0
    self.history = []

state = TrainState()
