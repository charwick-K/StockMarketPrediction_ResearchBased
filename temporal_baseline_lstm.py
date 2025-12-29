# =============================================================================
# Temporal Baseline LSTM
# Leak-Free Time-Series Forecasting with Risk-Aware Evaluation
#
# This model intentionally avoids architectural complexity to
# quantify the standalone capability of deep recurrence.
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

DATA_PATH = "data/stock.csv"   # <-- intentionally generic for reproducibility

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_global_seed(SEED)


# =============================================================================
# DATA PREPARATION
# =============================================================================
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Risk-aware transformation
df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df.dropna(inplace=True)

# Strict temporal hold-out
test_ratio = 0.15
split_idx = int(len(df) * (1 - test_ratio))

df_trainval = df.iloc[:split_idx]
df_test = df.iloc[split_idx:]


# =============================================================================
# DATASET
# =============================================================================
class TemporalReturnDataset(Dataset):
    """
    Converts a univariate return series into supervised
    sliding-window sequences without temporal leakage.
    """

    def __init__(self, frame: pd.DataFrame, window: int):
        self.returns = frame["log_return"].values.astype(np.float32)
        self.prices = frame["close"].values.astype(np.float32)
        self.window = window

    def __len__(self):
        return len(self.returns) - self.window

    def __getitem__(self, idx):
        x = self.returns[idx : idx + self.window]
        y = self.returns[idx + self.window]
        ref_price = self.prices[idx + self.window - 1]

        return (
            torch.tensor(x).unsqueeze(-1),
            torch.tensor(y),
            torch.tensor(ref_price),
        )


# =============================================================================
# MODEL
# =============================================================================
class TemporalBaselineLSTM(nn.Module):
    """
    Deep recurrent baseline used to establish
    a lower-bound performance reference.
    """

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=4,
            dropout=0.2,
            batch_first=True,
        )

        self.regressor = nn.Linear(128, 1)

    def forward(self, x):
        seq_out, _ = self.lstm(x)
        final_state = seq_out[:, -1, :]
        return self.regressor(final_state).squeeze(-1)


# =============================================================================
# CROSS-VALIDATION (FORWARD-CHAINING)
# =============================================================================
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

cv_metrics = []

print("\n===== Temporal Baseline LSTM | Cross-Validation =====\n")

for fold, (train_idx, val_idx) in enumerate(tscv.split(df_trainval), start=1):
    print(f"--- Fold {fold} ---")

    train_ds = TemporalReturnDataset(df_trainval.iloc[train_idx], WINDOW_SIZE)
    val_ds = TemporalReturnDataset(df_trainval.iloc[val_idx], WINDOW_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = TemporalBaselineLSTM().to(DEVICE)
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

    cv_metrics.append((rmse, mae, r2))

    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

# =============================================================================
# FINAL TRAINING (TRAIN + VAL)
# =============================================================================
print("\n===== Final Training =====")

final_loader = DataLoader(
    TemporalReturnDataset(df_trainval, WINDOW_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

final_model = TemporalBaselineLSTM().to(DEVICE)
optimizer = torch.optim.Adam(final_model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
    final_model.train()
    total_loss = 0
    for x, y, _ in final_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(final_model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(final_loader):.6f}")

# =============================================================================
# FINAL TEST (UNSEEN DATA)
# =============================================================================
print("\n===== Final Test (Unseen Data) =====")

test_loader = DataLoader(
    TemporalReturnDataset(df_test, WINDOW_SIZE),
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