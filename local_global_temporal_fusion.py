# =============================================================================
# Local–Global Temporal Fusion Network
# CNN for Short-Term Pattern Encoding + LSTM for Long-Term Memory
#
# Key Insight:
#   Financial time-series contain local micro-structures
#   (volatility bursts, momentum shifts) that recurrent
#   models alone often smooth out.
# =============================================================================

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =============================================================================
# CONFIGURATION
# =============================================================================
SEED = 42
WINDOW_SIZE = 30
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
N_SPLITS = 5

DATA_PATH = "data/stock.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =============================================================================
# DATA PREPARATION
# =============================================================================
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df.dropna(inplace=True)

test_ratio = 0.15
split_idx = int(len(df) * (1 - test_ratio))

df_trainval = df.iloc[:split_idx]
df_test = df.iloc[split_idx:]


# =============================================================================
# DATASET
# =============================================================================
class TemporalFusionDataset(Dataset):
    """
    Supplies short-term windows for convolutional encoding
    while preserving global temporal order.
    """

    def __init__(self, frame: pd.DataFrame, window: int):
        self.returns = frame["log_return"].values.astype(np.float32)
        self.prices = frame["close"].values.astype(np.float32)
        self.window = window

    def __len__(self):
        return len(self.returns) - self.window

    def __getitem__(self, idx):
        seq = self.returns[idx : idx + self.window]
        target = self.returns[idx + self.window]
        ref_price = self.prices[idx + self.window - 1]

        return (
            torch.tensor(seq).unsqueeze(0),  # (C=1, T)
            torch.tensor(target),
            torch.tensor(ref_price),
        )


# =============================================================================
# MODEL
# =============================================================================
class LocalGlobalTemporalFusion(nn.Module):
    """
    Local Encoder  : CNN captures short-range temporal motifs
    Global Encoder : LSTM integrates patterns over time
    """

    def __init__(self):
        super().__init__()

        # ---- Local Pattern Extractor ----
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # ---- Global Temporal Model ----
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
        )

        # ---- Output Layer ----
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, 1, T)
        local_features = self.conv_block(x)
        # (B, 64, T) ? (B, T, 64)
        local_features = local_features.permute(0, 2, 1)

        temporal_out, _ = self.lstm(local_features)
        final_state = temporal_out[:, -1, :]

        return self.output(final_state).squeeze(-1)


# =============================================================================
# CROSS-VALIDATION
# =============================================================================
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

print("\n===== Local–Global Temporal Fusion | CV =====\n")

for fold, (train_idx, val_idx) in enumerate(tscv.split(df_trainval), start=1):
    print(f"--- Fold {fold} ---")

    train_ds = TemporalFusionDataset(df_trainval.iloc[train_idx], WINDOW_SIZE)
    val_ds = TemporalFusionDataset(df_trainval.iloc[val_idx], WINDOW_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LocalGlobalTemporalFusion().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # -------- TRAIN --------
    for _ in range(EPOCHS):
        model.train()
        for x, y, _ in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

    # -------- VALIDATION --------
    model.eval()
    y_true, y_pred, refs = [], [], []

    with torch.no_grad():
        for x, y, ref in val_loader:
            preds = model(x.to(DEVICE)).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)
            refs.extend(ref.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    refs = np.array(refs)

    true_prices = refs * np.exp(y_true)
    pred_prices = refs * np.exp(y_pred)

    rmse = np.sqrt(mean_squared_error(true_prices, pred_prices))
    mae = mean_absolute_error(true_prices, pred_prices)
    r2 = r2_score(true_prices, pred_prices)

    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")


# =============================================================================
# FINAL TRAINING
# =============================================================================
print("\n===== Final Training =====")

final_loader = DataLoader(
    TemporalFusionDataset(df_trainval, WINDOW_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

final_model = LocalGlobalTemporalFusion().to(DEVICE)
optimizer = torch.optim.Adam(final_model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
    final_model.train()
    epoch_loss = 0.0
    for x, y, _ in final_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(final_model(x), y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1:03d} | Loss: {epoch_loss/len(final_loader):.6f}")


# =============================================================================
# FINAL TEST
# =============================================================================
print("\n===== Final Test (Unseen Data) =====")

test_loader = DataLoader(
    TemporalFusionDataset(df_test, WINDOW_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

final_model.eval()
y_true, y_pred, refs = [], [], []

with torch.no_grad():
    for x, y, ref in test_loader:
        preds = final_model(x.to(DEVICE)).cpu().numpy()
        y_true.extend(y.numpy())
        y_pred.extend(preds)
        refs.extend(ref.numpy())

true_prices = np.array(refs) * np.exp(np.array(y_true))
pred_prices = np.array(refs) * np.exp(np.array(y_pred))

print(f"RMSE: {np.sqrt(mean_squared_error(true_prices, pred_prices)):.4f}")
print(f"MAE : {mean_absolute_error(true_prices, pred_prices):.4f}")
print(f"R2  : {r2_score(true_prices, pred_prices):.4f}")