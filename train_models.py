import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib

# ========================
# CONFIG
# ========================
SEQ_LENGTH = 60
EPOCHS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# MODELS
# ========================

class GRUModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.gru = nn.GRU(input_size, 64, 2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out[:, :, :res.size(2)] + res)


class TCNModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.tcn = nn.Sequential(
            TemporalBlock(input_size, 32, 3, 1),
            TemporalBlock(32, 64, 3, 2)
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        return self.fc(out[:, :, -1])


# ========================
# LOAD DATA
# ========================
df = yf.download("NVDA", period="2y")

df['Return'] = df['Close'].pct_change()
df = df.dropna()

features = df[['Return', 'Open', 'High', 'Low', 'Volume']].values

# ========================
# SCALING
# ========================
scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)

joblib.dump(scaler, "scaler_v2.pkl")

# ========================
# CREATE SEQUENCES
# ========================
X, y = [], []

for i in range(len(scaled) - SEQ_LENGTH):
    X.append(scaled[i:i+SEQ_LENGTH])
    y.append(scaled[i+SEQ_LENGTH][0])  # predict Return

X = np.array(X)
y = np.array(y)

X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)

# ========================
# TRAIN FUNCTION
# ========================
def train_model(model, name):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()

        pred = model(X).squeeze()
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"{name} Epoch {epoch+1}, Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), name)
    print(f"{name} saved!")

# ========================
# TRAIN MODELS
# ========================
INPUT_SIZE = 5

train_model(GRUModel(INPUT_SIZE), "gru_model.pth")
train_model(TCNModel(INPUT_SIZE), "tcn_model.pth")