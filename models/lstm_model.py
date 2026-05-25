"""
LSTM model for option price / volatility prediction.
Uses sequential windows of features to capture temporal dynamics in IV surface.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import logging

log = logging.getLogger(__name__)

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_PATH = Path(__file__).parent / "lstm_model.pt"

SEQUENCE_LEN = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3


class OptionsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = SEQUENCE_LEN):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len]


class LSTMPricer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def load_data():
    X = pd.read_csv(PROC_DIR / "features.csv").values
    y = pd.read_csv(PROC_DIR / "targets.csv").squeeze().values
    return X, y


def train(X: np.ndarray, y: np.ndarray) -> LSTMPricer:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    train_ds = OptionsDataset(X_train, y_train)
    val_ds = OptionsDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMPricer(input_size=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(criterion(model(xb), yb).item())
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
        if (epoch + 1) % 10 == 0:
            log.info("Epoch %d/%d  val_loss=%.5f", epoch + 1, EPOCHS, val_loss)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    log.info("LSTM model saved -> %s", MODEL_PATH)
    return model


def evaluate(model: LSTMPricer, X: np.ndarray, y: np.ndarray) -> dict:
    device = next(model.parameters()).device
    ds = OptionsDataset(X, y)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in dl:
            preds.extend(model(xb.to(device)).cpu().numpy())
            targets.extend(yb.numpy())
    preds, targets = np.array(preds), np.array(targets)
    return {
        "rmse": float(np.sqrt(mean_squared_error(targets, preds))),
        "mae": float(mean_absolute_error(targets, preds)),
        "r2": float(r2_score(targets, preds)),
    }


def load_model(input_size: int) -> LSTMPricer:
    model = LSTMPricer(input_size=input_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    X, y = load_data()
    model = train(X, y)
    metrics = evaluate(model, X, y)
    log.info("LSTM test metrics: %s", metrics)
