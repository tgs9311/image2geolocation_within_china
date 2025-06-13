from model import ProvinceModel

import torch
from datasets import Dataset
import os
import json
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score
from tqdm import tqdm
from torch import Tensor
import math
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load country mapping
with open("shape_centers_1.json", "r") as f:
    shape_centers = json.load(f)

# Initialize model
num_classes = len(shape_centers)
print(f'Num Classes: {num_classes}')
model = ProvinceModel(num_classes=num_classes)
model.to(device)

centers = []
for i in range(num_classes):
    centers.append(shape_centers[i]['center'])
center_tensor = Tensor(centers)
center_tensor = center_tensor.transpose(0, 1).to(device)

rad_torch = torch.tensor(6378137.0, dtype=torch.float64)


def haversine_distance(pred_coords, true_coords):
    R = 6371  # km
    pred_rad = torch.deg2rad(pred_coords)
    true_rad = torch.deg2rad(true_coords)
    dlat = pred_rad[:, 0] - true_rad[:, 0]
    dlon = pred_rad[:, 1] - true_rad[:, 1]
    a = torch.sin(dlat / 2) ** 2 + torch.cos(pred_rad[:, 0]) * torch.cos(true_rad[:, 0]) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.arcsin(torch.sqrt(a))
    return R * c

def haversine_matrix(x: Tensor, y: Tensor) -> Tensor:
    x_rad, y_rad = torch.deg2rad(x), torch.deg2rad(y)
    delta = x_rad.unsqueeze(2) - y_rad
    p = torch.cos(x_rad[:, 1]).unsqueeze(1) * torch.cos(y_rad[1, :]).unsqueeze(0)
    a = torch.sin(delta[:, 1, :] / 2)**2 + p * torch.sin(delta[:, 0, :] / 2)**2
    c = 2 * torch.arcsin(torch.sqrt(a))
    km = (rad_torch * c) / 1000
    return km

tau = 75

def HaversineLoss(probs, loc, tau=75):
    distances = haversine_matrix(loc, center_tensor)
    weights = torch.exp(-distances / tau)
    loss = (-probs.log() * weights).sum(dim=-1).mean()
    label = distances.argmin(dim=-1)
    return loss, label

def HaversineCoordLoss(probs, true_coords):
    pred_coords = probs @ center_tensor.T
    distances = haversine_distance(pred_coords, true_coords)
    return distances.mean()

def MixedLoss(logits, probs, loc, alpha = 0.5):
    loss_coord,label = HaversineLoss(probs, loc)
    loss_cls = F.cross_entropy(logits, label)
    loss = loss_coord + alpha * loss_cls
    return loss,label

# Loss and optimizer
criterion = HaversineLoss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-4)

# Load dataset
full_dataset = Dataset.load_from_disk("/root/autodl-tmp/TUXUN/dataset_one")

split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']


# Custom collate function
def collate_fn(batch):
    batch = [b for b in batch if b is not None and b['embedding'] is not None]
    if len(batch) == 0:
        return None
    
    # Stack embeddings
    embeddings = torch.stack([torch.tensor(b['embedding']) for b in batch])
    lat = torch.stack([torch.tensor(b['latitude']) for b in batch])
    lon = torch.stack([torch.tensor(b['longitude']) for b in batch])
    
    # Convert country names to indices and create one-hot vectors
    return {
        'embedding': embeddings,
        'loc': torch.stack([lat, lon], dim=-1)
    }

# Create DataLoaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    collate_fn=collate_fn
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,
    collate_fn=collate_fn
)

def evaluate(model, dataloader):
    model.eval()
    val_loss = 0
    val_acc = 0
    processed_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
                
            images = batch['embedding'].to(device)
            loc = batch['loc'].to(device)
            
            logits = model(images, return_logits=True)
            outputs = F.softmax(logits, dim=-1)
            loss,label = MixedLoss(logits,outputs,loc)
            val_acc += (outputs.max(dim=-1).indices == label.reshape(-1)).sum().item()
            processed_samples += len(loc)
            val_loss += loss.item() 
    
    return val_loss / processed_samples, val_acc / processed_samples

# Training loop
num_epochs = 50
best_val_loss = float('inf')
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=num_epochs,
    eta_min=1e-5
)

best_val_acc = 0
best_val_loss = 1000000000
for epoch in tqdm(range(num_epochs)):
    # Training phase
    model.train()
    total_loss = 0
    train_acc = 0
    processed_samples = 0
    
    for batch in tqdm(train_dataloader):
        if batch is None:
            continue
            
        images = batch['embedding'].to(device)
        loc = batch['loc'].to(device)
        
        optimizer.zero_grad()
        logits = model(images, return_logits=True)
        outputs = F.softmax(logits, dim=-1)
        loss,label = MixedLoss(logits,outputs,loc)
        train_acc += (outputs.max(dim=-1).indices == label.reshape(-1)).sum().item()

        loss.backward()
        optimizer.step()
        
        batch_size = len(loc)
        total_loss += loss.item() 
        processed_samples += batch_size
    
    scheduler.step()
    # Print training stats
    if processed_samples > 0:
        avg_train_loss = total_loss / processed_samples
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Processed samples: {processed_samples}")
        print(f"Accuracy: {train_acc / processed_samples:.4f}")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")
    else:
        print(f"\nEpoch {epoch+1}/{num_epochs}: No training samples processed")
        continue
    
    # Validation phase
    val_loss, val_acc = evaluate(model, val_dataloader)
    
    print(f"Val Metrics:")
    print(f"Valod Loss:{val_loss:.4f},Accuracy: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "./province_2.pth")
        print("Saved new best model")
