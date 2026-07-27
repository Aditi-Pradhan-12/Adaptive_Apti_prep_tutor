"""
non_verbal_cnn.py

A genuinely trained deep learning model (proposal section 5.5): a small
multi-task CNN with two output heads sharing a convolutional trunk:
  - mirror_head: binary classification (normal vs mirrored)
  - rotation_head: 4-way classification (0 / 90 / 180 / 270 degrees)

This is the ONE component in the whole project that is actually trained
via backpropagation on labeled data, as opposed to the rule-based /
retrieval-based logic elsewhere -- worth highlighting explicitly in your
report and viva as "the trained DL model."

Run this file directly to train from scratch and save weights:
    python non_verbal_cnn.py
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from non_verbal_dataset import generate_dataset, IMG_SIZE

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "non_verbal_cnn.pt")


class NonVerbalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32 -> 16
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16 -> 8
        )
        self.shared_fc = nn.Sequential(nn.Linear(32 * 8 * 8, 64), nn.ReLU())
        self.mirror_head = nn.Linear(64, 2)
        self.rotation_head = nn.Linear(64, 4)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.shared_fc(x)
        return self.mirror_head(x), self.rotation_head(x)


def train(n_train=6000, n_test=1200, epochs=10, batch_size=64, lr=1e-3, save_path=MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_imgs, train_mirror, train_rot = generate_dataset(n_train, jitter=True)
    test_imgs, test_mirror, test_rot = generate_dataset(n_test, jitter=True)

    def to_loader(imgs, mirror, rot, shuffle):
        X = torch.tensor(imgs).unsqueeze(1)  # add channel dim -> (N,1,H,W)
        ym = torch.tensor(mirror)
        yr = torch.tensor(rot)
        return DataLoader(TensorDataset(X, ym, yr), batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(train_imgs, train_mirror, train_rot, shuffle=True)
    test_loader = to_loader(test_imgs, test_mirror, test_rot, shuffle=False)

    model = NonVerbalCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X, ym, yr in train_loader:
            X, ym, yr = X.to(device), ym.to(device), yr.to(device)
            optimizer.zero_grad()
            pred_mirror, pred_rot = model(X)
            loss = criterion(pred_mirror, ym) + criterion(pred_rot, yr)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)

        avg_loss = total_loss / n_train

        model.eval()
        correct_mirror, correct_rot, total = 0, 0, 0
        with torch.no_grad():
            for X, ym, yr in test_loader:
                X, ym, yr = X.to(device), ym.to(device), yr.to(device)
                pred_mirror, pred_rot = model(X)
                correct_mirror += (pred_mirror.argmax(1) == ym).sum().item()
                correct_rot += (pred_rot.argmax(1) == yr).sum().item()
                total += X.size(0)

        print(f"Epoch {epoch:2d}/{epochs} | loss {avg_loss:.4f} | "
              f"mirror_acc {correct_mirror/total:.4f} | rotation_acc {correct_rot/total:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved trained weights -> {save_path}")
    return model


def load_model(path: str = MODEL_PATH) -> NonVerbalCNN:
    model = NonVerbalCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


if __name__ == "__main__":
    train()
