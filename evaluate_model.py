import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========================
# USER INPUT
# ========================
print("Select Model:")
print("1 - LSTM")
print("2 - GRU")
print("3 - TCN")
print("4 - Compare All Models")

choice = int(input("Enter choice (1/2/3/4): "))

if choice == 1:
    MODEL_TYPE = "LSTM"
elif choice == 2:
    MODEL_TYPE = "GRU"
elif choice == 3:
    MODEL_TYPE = "TCN"
elif choice == 4:
    MODEL_TYPE = "COMPARE"
else:
    print("Invalid choice")
    exit()

# ========================
# CONFIG
# ========================
DATA_PATH = r"C:\Users\mohan\Downloads\finance (1)\finance\nvidia.csv"
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
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

close_prices = df['Close'].values

df['Return'] = df['Close'].pct_change()
df = df.dropna()

data = df[['Return', 'Open', 'High', 'Low', 'Volume']].values

# ========================
# SCALER
# ========================
scaler = joblib.load("scaler_v2.pkl")

split = int(len(data) * 0.8)
test_data = data[split:]
test_scaled = scaler.transform(test_data)

# ========================
# SEQUENCES
# ========================
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][0])
    return np.array(X), np.array(y)

X_test, y_test = create_sequences(test_scaled, SEQ_LENGTH)

# ========================
# EVALUATION FUNCTION
# ========================
def evaluate_model(model, model_name):
    model = model.to(device)
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_returns = model(X_tensor).cpu().numpy().flatten()

    # inverse scale prediction
    dummy_pred = np.zeros((len(pred_returns), scaler.n_features_in_))
    dummy_pred[:, 0] = pred_returns
    pred_returns_inv = scaler.inverse_transform(dummy_pred)[:, 0]

    # inverse scale actual
    dummy_actual = np.zeros((len(y_test), scaler.n_features_in_))
    dummy_actual[:, 0] = y_test
    actual_returns_inv = scaler.inverse_transform(dummy_actual)[:, 0]

    # convert returns to prices
    close_prices_adj = close_prices[1:]
    close_test = close_prices_adj[-len(actual_returns_inv):]

    pred_prices = close_test * (1 + pred_returns_inv)
    actual_prices = close_test * (1 + actual_returns_inv)

    # metrics
    rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
    mae = mean_absolute_error(actual_prices, pred_prices)
    r2 = r2_score(actual_prices, pred_prices)
    mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100

    direction = np.mean(
        (np.diff(actual_prices) > 0) ==
        (np.diff(pred_prices) > 0)
    )
    
    # Additional metrics
    cum_ret_actual = (actual_prices[-1] / actual_prices[0]) - 1
    cum_ret_pred = (pred_prices[-1] / pred_prices[0]) - 1
    
    sharpe_actual = np.mean(actual_returns_inv) / np.std(actual_returns_inv) * np.sqrt(252)
    
    def max_drawdown(prices):
        cum_returns = (prices / prices[0]) - 1
        peak = np.maximum.accumulate(cum_returns)
        return np.min(cum_returns - peak)
    
    mdd_actual = max_drawdown(actual_prices)
    mdd_pred = max_drawdown(pred_prices)

    return {
        "name": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "direction": direction,
        "actual_prices": actual_prices,
        "pred_prices": pred_prices,
        "cum_ret_actual": cum_ret_actual,
        "cum_ret_pred": cum_ret_pred,
        "sharpe_actual": sharpe_actual,
        "mdd_actual": mdd_actual,
        "mdd_pred": mdd_pred,
        "actual_returns": actual_returns_inv,
        "pred_returns": pred_returns_inv
    }

# ========================
# VISUALIZATION FUNCTIONS
# ========================
def plot_results(result):
    actual_prices = result["actual_prices"]
    pred_prices = result["pred_prices"]
    
    # 1. Actual vs Predicted
    plt.figure(figsize=(12,6))
    plt.plot(actual_prices, label="Actual Price")
    plt.plot(pred_prices, label="Predicted Price")
    plt.title(f"{result['name']} - Actual vs Predicted Stock Prices")
    plt.legend()
    plt.show()
    
    # 2. Error Plot
    errors = actual_prices - pred_prices
    plt.figure(figsize=(10,5))
    plt.plot(errors)
    plt.title(f"{result['name']} - Prediction Error")
    plt.ylabel("Error")
    plt.xlabel("Time")
    plt.show()
    
    # 3. Scatter Plot
    plt.figure(figsize=(6,6))
    plt.scatter(actual_prices, pred_prices, alpha=0.5)
    plt.plot([actual_prices.min(), actual_prices.max()], 
             [actual_prices.min(), actual_prices.max()], 'r--', label="Perfect Prediction")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{result['name']} - Actual vs Predicted Scatter")
    plt.legend()
    plt.show()
    
    # 4. Direction Plot
    actual_dirs = np.diff(actual_prices) > 0
    pred_dirs = np.diff(pred_prices) > 0
    correct = actual_dirs == pred_dirs
    
    plt.figure(figsize=(12,6))
    plt.plot(actual_prices[1:], label="Actual Price")
    
    correct_idx = np.where(correct)[0]
    wrong_idx = np.where(~correct)[0]
    
    plt.scatter(correct_idx + 1, actual_prices[correct_idx + 1], 
               color='green', s=10, label='Correct Direction')
    plt.scatter(wrong_idx + 1, actual_prices[wrong_idx + 1], 
               color='red', s=10, label='Wrong Direction')
    
    plt.title(f"{result['name']} - Direction Prediction")
    plt.legend()
    plt.show()
    
    # 5. Cumulative Return Plot
    cum_actual = (actual_prices / actual_prices[0]) - 1
    cum_pred = (pred_prices / pred_prices[0]) - 1
    
    plt.figure(figsize=(12,6))
    plt.plot(cum_actual, label="Actual", linewidth=2)
    plt.plot(cum_pred, label="Predicted", linewidth=2)
    plt.title(f"{result['name']} - Cumulative Returns")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 6. Drawdown Plot
    def drawdown_curve(prices):
        cum = (prices / prices[0]) - 1
        peak = np.maximum.accumulate(cum)
        return cum - peak
    
    dd_actual = drawdown_curve(actual_prices)
    dd_pred = drawdown_curve(pred_prices)
    
    plt.figure(figsize=(12,6))
    plt.plot(dd_actual, label="Actual", linewidth=2)
    plt.plot(dd_pred, label="Predicted", linewidth=2)
    plt.title(f"{result['name']} - Drawdown Curve")
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_comparison(results):
    plt.figure(figsize=(14,6))
    
    # Plot actual prices
    plt.plot(results[0]["actual_prices"], label="Actual", linewidth=2, color='black')
    
    # Plot predictions for each model
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for i, res in enumerate(results):
        plt.plot(res["pred_prices"], label=res["name"], 
                linewidth=1.5, color=colors[i % len(colors)], alpha=0.7)
    
    plt.title("Model Comparison - Actual vs Predicted Prices")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Bar chart for metrics comparison
    metrics = ['rmse', 'mae', 'mape', 'direction']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        values = [res[metric] for res in results]
        names = [res["name"] for res in results]
        
        axes[idx].bar(names, values, color=['blue', 'green', 'red'])
        axes[idx].set_title(f'{metric.upper()} Comparison')
        axes[idx].set_ylabel(metric.upper())
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ========================
# SINGLE MODEL
# ========================
if MODEL_TYPE != "COMPARE":
    input_size = data.shape[1]
    
    if MODEL_TYPE == "LSTM":
        model = LSTMModel(input_size)
        model.load_state_dict(torch.load("lstm_model_v2.pth", map_location=device))

    elif MODEL_TYPE == "GRU":
        model = GRUModel(input_size)
        model.load_state_dict(torch.load("gru_model.pth", map_location=device))

    elif MODEL_TYPE == "TCN":
        model = TCNModel(input_size)
        model.load_state_dict(torch.load("tcn_model.pth", map_location=device))

    result = evaluate_model(model, MODEL_TYPE)

    # Print results
    print("\n" + "="*50)
    print(f"MODEL: {result['name']}")
    print("="*50)
    print(f"RMSE:  {result['rmse']:.4f}")
    print(f"MAE:   {result['mae']:.4f}")
    print(f"R2:    {result['r2']:.4f}")
    print(f"MAPE:  {result['mape']:.2f}%")
    print(f"Direction Accuracy: {result['direction']:.2%}")
    print("-"*50)
    print(f"Cumulative Return Actual:   {result['cum_ret_actual']:.2%}")
    print(f"Cumulative Return Predicted: {result['cum_ret_pred']:.2%}")
    print(f"Sharpe Ratio (Actual):      {result['sharpe_actual']:.2f}")
    print(f"Max Drawdown Actual:        {result['mdd_actual']:.2%}")
    print(f"Max Drawdown Predicted:     {result['mdd_pred']:.2%}")
    print("="*50)
    
    # Plot all visualizations
    plot_results(result)

# ========================
# COMPARE ALL MODELS
# ========================
else:
    input_size = data.shape[1]
    models = []

    lstm = LSTMModel(input_size)
    lstm.load_state_dict(torch.load("lstm_model_v2.pth", map_location=device))
    models.append(("LSTM", lstm))

    gru = GRUModel(input_size)
    gru.load_state_dict(torch.load("gru_model.pth", map_location=device))
    models.append(("GRU", gru))

    tcn = TCNModel(input_size)
    tcn.load_state_dict(torch.load("tcn_model.pth", map_location=device))
    models.append(("TCN", tcn))

    results = []

    for name, model in models:
        res = evaluate_model(model, name)
        results.append(res)

    # Print comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    for res in results:
        print(f"\n{res['name']}:")
        print(f"  RMSE:    {res['rmse']:.4f}")
        print(f"  MAE:     {res['mae']:.4f}")
        print(f"  R2:      {res['r2']:.4f}")
        print(f"  MAPE:    {res['mape']:.2f}%")
        print(f"  Direction: {res['direction']:.2%}")
        print(f"  Cum Return: {res['cum_ret_pred']:.2%}")
    
    # Find best model for each metric
    print("\n" + "-"*60)
    print("BEST PERFORMING MODELS:")
    print("-"*60)
    
    metrics_to_check = ['rmse', 'mae', 'r2', 'direction', 'mape']
    for metric in metrics_to_check:
        if metric in ['rmse', 'mae', 'mape']:
            best_model = min(results, key=lambda x: x[metric])
        else:
            best_model = max(results, key=lambda x: x[metric])
        print(f"Best {metric.upper()}: {best_model['name']} ({best_model[metric]:.4f})")
    
    print("="*60)
    
    # Plot comparison visualizations
    plot_comparison(results)
    
    # Plot individual results for each model
    for res in results:
        plot_results(res)