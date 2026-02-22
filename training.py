import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import matplotlib.pyplot as plt

from utils.dataset import MultiLabelDataset
from utils.augmentations import get_transforms
from utils.helpers import MaskedBCELoss

def compute_pos_weights(label_file):
    pos = [0, 0, 0, 0]     # positive counts
    total = [0, 0, 0, 0]   # non-NA counts

    with open(label_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        # Format: image.jpg attr1 attr2 attr3 attr4
        a1, a2, a3, a4 = parts[1], parts[2], parts[3], parts[4]
        labels = [a1, a2, a3, a4]

        for i, v in enumerate(labels):
            if v != "NA":
                total[i] += 1
                if v == "1":
                    pos[i] += 1

    weights = []
    for i in range(4):
        if pos[i] == 0:
            weights.append(1.0)
        else:
            neg = total[i] - pos[i]
            weights.append(neg / pos[i])

    return torch.tensor(weights).float()


# --------------------------------------------------------------------
# TRAINING FUNCTION
# --------------------------------------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_file = "data/labels.txt"
    img_dir = "data/images"

    # Dataset + DataLoader
    dataset = MultiLabelDataset(label_file, img_dir, transform=get_transforms())
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Compute imbalance weights
    pos_weights = compute_pos_weights(label_file).to(device)

    # Load pretrained ResNet50
    model = resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 4),
        nn.Sigmoid()
    )
    model.to(device)

    # Loss + Optimizer
    loss_fn = MaskedBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_log = []
    iteration = 0

    # ---------------- TRAIN LOOP -----------------
    for epoch in range(5):
        for imgs, labels, mask in loader:
            imgs, labels, mask = imgs.to(device), labels.to(device), mask.to(device)

            optimizer.zero_grad()
            preds = model(imgs)

            loss = loss_fn(preds, labels, mask)
            loss.backward()
            optimizer.step()

            iteration += 1
            loss_log.append(loss.item())

            print(f"Epoch {epoch} | Iter {iteration} | Loss {loss.item():.4f}")

    # Save model + loss log
    torch.save(model.state_dict(), "models/checkpoint.pth")
    torch.save({"iter": list(range(len(loss_log))), "loss": loss_log},
               "models/loss_log.pt")

    print("\n✔ Training complete! Model saved at models/checkpoint.pth\n")


# --------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------
if __name__ == "__main__":
    train()