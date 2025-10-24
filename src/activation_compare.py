import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# ---- Activation Functions ----
activations = {
    "ReLU": nn.ReLU(inplace=True),
    "LeakyReLU": nn.LeakyReLU(0.01, inplace=True),
    "ELU": nn.ELU(alpha=1.0, inplace=True),
}

# ---- Model Definition ----
class SmallCNN(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            activation,
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            activation,
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            activation,
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ---- Data ----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
train_data, val_data = random_split(dataset, [55000, 5000])
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

# ---- Training Setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10
lr = 0.01
loss_fn = nn.CrossEntropyLoss()

# ---- Storage for all results ----
train_losses = {}
train_accuracies = {}
val_losses = {}
val_errors = {}
val_accuracies = {}

for name, act in activations.items():
    print(f"\n=== Training with {name} ===")
    model = SmallCNN(act).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    loss_list = []
    acc_list = []
    val_loss_list = []
    val_error_list = []
    val_acc_list = []

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        train_accuracy = correct / total
        loss_list.append(train_loss)
        acc_list.append(train_accuracy)

        # ---- Validate ----
        model.eval()
        val_loss_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = loss_fn(out, y)
                val_loss_sum += loss.item() * x.size(0)
                preds = out.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        val_err = 1.0 - val_acc
        val_loss = val_loss_sum / len(val_loader.dataset)

        val_loss_list.append(val_loss)
        val_error_list.append(val_err)
        val_acc_list.append(val_acc)

        print(f"Epoch {epoch:02d} | Train Loss={train_loss:.4f} | Train Acc={train_accuracy:.4f} | "
              f"Val Loss={val_loss:.4f} | Val Err={val_err:.4f} | Val Acc={val_acc:.4f}")

    # Store results
    train_losses[name] = loss_list
    train_accuracies[name] = acc_list
    val_losses[name] = val_loss_list
    val_errors[name] = val_error_list
    val_accuracies[name] = val_acc_list

# ---- Plot 1: Training Loss ----
epochs_list = list(range(1, epochs + 1))
plt.figure(figsize=(8,5))
for name in activations.keys():
    plt.plot(epochs_list, train_losses[name], label=f"Train Loss ({name})")
plt.title("Training Loss Comparison (ReLU vs LeakyReLU vs ELU)")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("training_loss_comparison.png", dpi=200)
plt.show()

# ---- Plot 2: Training Accuracy ----
plt.figure(figsize=(8,5))
for name in activations.keys():
    plt.plot(epochs_list, train_accuracies[name], label=f"Train Acc ({name})")
plt.title("Training Accuracy Comparison (ReLU vs LeakyReLU vs ELU)")
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("training_accuracy_comparison.png", dpi=200)
plt.show()

# ---- Plot 3: Validation Error ----
plt.figure(figsize=(8,5))
for name in activations.keys():
    plt.plot(epochs_list, val_errors[name], label=f"Val Error ({name})")
plt.title("Validation Error Comparison (ReLU vs LeakyReLU vs ELU)")
plt.xlabel("Epoch")
plt.ylabel("Validation Error (1 - Accuracy)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("validation_error_comparison.png", dpi=200)
plt.show()

# ---- Plot 4: Validation Accuracy ----
plt.figure(figsize=(8,5))
for name in activations.keys():
    plt.plot(epochs_list, val_accuracies[name], label=f"Val Acc ({name})")
plt.title("Validation Accuracy Comparison (ReLU vs LeakyReLU vs ELU)")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("validation_accuracy_comparison.png", dpi=200)
plt.show()
