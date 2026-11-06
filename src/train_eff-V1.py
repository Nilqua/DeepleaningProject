import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ==========================================
# 1. Data Preprocessing & DataLoader
# ==========================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder("data/train", transform=train_transform)
val_dataset   = datasets.ImageFolder("data/val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ==========================================
# 2. Model Setup - Baseline
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_v2_s(weights="DEFAULT")

# Freeze Backbone ทั้งหมดเพื่อเทรนเฉพาะส่วนหัว
for param in model.features.parameters():
    param.requires_grad = False

# เปลี่ยน Classifier ท้ายสุดเป็น 12 คลาส
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_features=1280, out_features=12)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# ==========================================
# 3. Training & Validation Loop (พื้นฐาน)
# ==========================================
for epoch in range(10):
    # --- Train ---
    model.train()
    train_loss, train_correct = 0.0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()

    # --- Validation ---
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    print(f"Epoch {epoch+1:02d} | "
          f"Train Loss: {train_loss/len(train_dataset):.4f} Acc: {train_correct/len(train_dataset):.4f} | "
          f"Val Loss: {val_loss/len(val_dataset):.4f} Acc: {val_correct/len(val_dataset):.4f}")