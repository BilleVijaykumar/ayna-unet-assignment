import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import UNet
from dataset import PolygonDataset
from utils import transform, color2id, save_model
import wandb

wandb.init(project="ayna-unet")

train_dataset = PolygonDataset("dataset/training", transform=transform, color2id=color2id)
val_dataset = PolygonDataset("dataset/validation", transform=transform, color2id=color2id)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

model = UNet(in_channels=3, n_classes=3).cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    total_loss = 0
    for img, color_id, target in train_loader:
        img, color_id, target = img.cuda(), color_id.cuda(), target.cuda()
        output = model(img, color_id)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    wandb.log({"train_loss": total_loss / len(train_loader)})

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for img, color_id, target in val_loader:
            img, color_id, target = img.cuda(), color_id.cuda(), target.cuda()
            output = model(img, color_id)
            loss = criterion(output, target)
            val_loss += loss.item()
        wandb.log({"val_loss": val_loss / len(val_loader)})

save_model(model, "unet_colorizer.pth")