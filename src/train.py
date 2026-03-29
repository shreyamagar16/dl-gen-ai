import torch
from torch.utils.data import DataLoader
import pandas as pd
import os

from dataset import AudioDataset
from models import CNNModel
import wandb

# INIT
wandb.init(project="messy-mashup")

BASE = "/kaggle/input/competitions/jan-2026-dl-gen-ai-project/messy_mashup"

# SIMPLE DATA (we’ll improve later)
file_paths = []
labels = []

genres = ["blues","classical","country","disco","hiphop",
          "jazz","metal","pop","reggae","rock"]

label_map = {g:i for i,g in enumerate(genres)}

# Load training data
for genre in genres:
    genre_path = os.path.join(BASE, "genres_stems", genre)
    
    for folder in os.listdir(genre_path):
        full_path = os.path.join(genre_path, folder)
        
        if os.path.isdir(full_path):
            # use drums.wav as sample (simple baseline)
            audio_file = os.path.join(full_path, "drums.wav")
            
            if os.path.exists(audio_file):
                file_paths.append(audio_file)
                labels.append(label_map[genre])

# Dataset
dataset = AudioDataset(file_paths, labels)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model
model = CNNModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# TRAIN
for epoch in range(3):
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        
        loss.backward()
        optimizer.step()
    
    wandb.log({"loss": loss.item()})
    print(f"Epoch {epoch} Loss: {loss.item()}")
