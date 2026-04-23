import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
import joblib
import pandas as pd

# ========================
# CONFIG
# ========================
SEQ_LENGTH = 60
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# MODELS
# ========================

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, 2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


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
            TemporalBlock(input_size, 32, 3, dilation=1),
            TemporalBlock(32, 64, 3, dilation=2)
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        return self.fc(out[:, :, -1])


# ========================
# UI
# ========================
st.title("📈 Stock Price Predictor")

model_choice = st.selectbox("Select Model", ["LSTM", "GRU", "TCN"])

stock = st.text_input("Enter Stock Symbol", "NVDA").strip().split(',')[0]

# ========================
# LOAD MODEL
# ========================
INPUT_SIZE = 5

if model_choice == "LSTM":
    model = LSTMModel(INPUT_SIZE)
    model.load_state_dict(torch.load("lstm_model_v2.pth", map_location=device))

elif model_choice == "GRU":
    model = GRUModel(INPUT_SIZE)
    model.load_state_dict(torch.load("gru_model.pth", map_location=device))

elif model_choice == "TCN":
    model = TCNModel(INPUT_SIZE)
    model.load_state_dict(torch.load("tcn_model.pth", map_location=device))

model = model.to(device)
model.eval()

scaler = joblib.load("scaler_v2.pkl")

# ========================
# PREDICTION
# ========================
if st.button("Predict Next Day Price"):

    df = yf.download(stock, period="120d")

    if df.empty:
        st.error("Failed to fetch stock data.")
    elif len(df) < SEQ_LENGTH + 1:
        st.error("Not enough data!")
    else:
        # FIX: flatten columns (important for yfinance multi-index)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # ========================
        # FEATURE ENGINEERING
        # ========================
        df['Return'] = df['Close'].pct_change()
        df = df.dropna()

        features = df[['Return', 'Open', 'High', 'Low', 'Volume']]
        recent = features.tail(SEQ_LENGTH).values

        # ========================
        # SCALE
        # ========================
        scaled = scaler.transform(recent)

        # ========================
        # MODEL INPUT
        # ========================
        X = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)

        # ========================
        # PREDICT
        # ========================
        with torch.no_grad():
            pred_return = model(X).cpu().numpy().flatten()[0]

        # ========================
        # INVERSE SCALE (SAFE)
        # ========================
        dummy = np.zeros((1, scaler.n_features_in_))
        dummy[0, 0] = pred_return

        pred_return = float(scaler.inverse_transform(dummy)[0][0])

        # ========================
        # SAFE LAST PRICE FIX ✅
        # ========================
        last_price = float(df['Close'].values[-1])

        pred_price = last_price * (1 + pred_return)

        # ========================
        # NEXT TRADING DAY
        # ========================
        today = pd.Timestamp.today()
        next_date = today + pd.Timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += pd.Timedelta(days=1)

        # ========================
        # DISPLAY
        # ========================
        st.success(f"Predicted Price ({next_date.date()}): ${pred_price:.2f}")
        st.write(f"Predicted Return: {pred_return:.6f}")

        st.line_chart(df['Close'])