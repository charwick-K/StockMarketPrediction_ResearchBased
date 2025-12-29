# =============================================================================
# Interpretable Temporal Attention Network
# LSTM + Attribution-Oriented Attention (Leak-Free)
#
# Core Idea:
#   Attention weights are treated as temporal attribution scores,
#   enabling post-hoc interpretability of model decisions.
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
EPOCHS = 25
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
class TemporalAttributionDataset(Dataset):
    """
    Provides sequences suitable for attribution-based attention
    while preserving strict temporal order.
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
            torch.tensor(seq).unsqueeze(-1),
            torch.tensor(target),
            torch.tensor(ref_price),
        )


# =============================================================================
# ATTENTION MODULE (ATTRIBUTION-ORIENTED)
# =============================================================================
class TemporalAttributionAttention(nn.Module):
    """
    Learns temporal importance scores that can be interpreted
    as contribution strength of each timestep.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.energy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lstm_outputs):
        scores = self.energy(lstm_outputs)              # (B, T, 1)
        weights = torch.softmax(scores, dim=1)          # temporal attribution
        context = torch.sum(weights * lstm_outputs, 1)  # weighted sum
        return context, weights


# =============================================================================
# MODEL
# =============================================================================
class InterpretableTemporalAttention(nn.Module):
    """
    Combines temporal memory with attribution-aware attention.
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=3,
            dropout=0.2,
            batch_first=True,
        )

        self.attention = TemporalAttributionAttention(128)
        self.regressor = nn.Linear(128, 1)

    def forward(self, x):
        lstm_out, _ = self.encoder(x)
        context, attn_weights = self.attention(lstm_out)
        prediction = self.regressor(context).squeeze(-1)
        return prediction, attn_weights


# =============================================================================
# CROSS-VALIDATION
# =============================================================================
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

print("\n===== Interpretable Temporal Attention | CV =====\n")

for fold, (train_idx, val_idx) in enumerate(tscv.split(df_trainval), start=1):
    print(f"--- Fold {fold} ---")

    train_ds = TemporalAttributionDataset(df_trainval.iloc[train_idx], WINDOW_SIZE)
    val_ds = TemporalAttributionDataset(df_trainval.iloc[val_idx], WINDOW_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = InterpretableTemporalAttention().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # -------- TRAIN --------
    for _ in range(EPOCHS):
        model.train()
        for x, y, _ in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            preds, _ = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

    # -------- VALIDATION --------
    model.eval()
    y_true, y_pred, refs = [], [], []

    with torch.no_grad():
        for x, y, ref in val_loader:
            preds, _ = model(x.to(DEVICE))
            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())
            refs.extend(ref.numpy())

    true_prices = np.array(refs) * np.exp(np.array(y_true))
    pred_prices = np.array(refs) * np.exp(np.array(y_pred))

    rmse = np.sqrt(mean_squared_error(true_prices, pred_prices))
    mae = mean_absolute_error(true_prices, pred_prices)
    r2 = r2_score(true_prices, pred_prices)

    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")


# =============================================================================
# FINAL TRAINING
# =============================================================================
print("\n===== Final Training =====")

final_loader = DataLoader(
    TemporalAttributionDataset(df_trainval, WINDOW_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

final_model = InterpretableTemporalAttention().to(DEVICE)
optimizer = torch.optim.Adam(final_model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
    final_model.train()
    epoch_loss = 0.0
    for x, y, _ in final_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        preds, _ = final_model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1:03d} | Loss: {epoch_loss/len(final_loader):.6f}")


# =============================================================================
# FINAL TEST
# =============================================================================
print("\n===== Final Test (Unseen Data) =====")

test_loader = DataLoader(
    TemporalAttributionDataset(df_test, WINDOW_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

final_model.eval()
y_true, y_pred, refs = [], [], []
all_attn = []

with torch.no_grad():
    for x, y, ref in test_loader:
        preds, attn = final_model(x.to(DEVICE))
        y_true.extend(y.numpy())
        y_pred.extend(preds.cpu().numpy())
        refs.extend(ref.numpy())
        all_attn.append(attn.cpu().numpy())

true_prices = np.array(refs) * np.exp(np.array(y_true))
pred_prices = np.array(refs) * np.exp(np.array(y_pred))

print(f"RMSE: {np.sqrt(mean_squared_error(true_prices, pred_prices)):.4f}")
print(f"MAE : {mean_absolute_error(true_prices, pred_prices):.4f}")
print(f"R2  : {r2_score(true_prices, pred_prices):.4f}")

# Attention weights (all_attn) can be visualized post-hoc