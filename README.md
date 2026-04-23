# Stock Price Predictor

This project is a Streamlit web app for predicting the next day's stock price using deep learning models (LSTM, GRU, TCN) trained on historical stock data. It uses PyTorch, yfinance, and scikit-learn for model inference and data processing.

## Features
- Predicts next-day price for any stock symbol (default: NVDA)
- Supports LSTM, GRU, and TCN models
- Interactive UI with Streamlit
- Uses pre-trained models and a scaler

## Requirements
- Python 3.8+
- pip

### Python Packages
- streamlit
- torch
- numpy
- pandas
- yfinance
- joblib

## Setup
1. **Clone or download this repository.**
2. **Install dependencies:**
   ```bash
   pip install streamlit torch numpy pandas yfinance joblib
   ```
3. **Ensure the following files are present in the project directory:**
   - `app.py` (main Streamlit app)
   - `lstm_model_v2.pth`, `gru_model.pth`, `tcn_model.pth` (pre-trained models)
   - `scaler_v2.pkl` (scaler for feature normalization)

## Running the Streamlit App
```bash
streamlit run app.py
```
- Open the provided local URL in your browser.
- Enter a stock symbol (e.g., NVDA, AAPL, MSFT) and select a model.
- Click "Predict Next Day Price" to see the prediction and chart.

## Training Models (Optional)
If you want to retrain the models:
```bash
python train_models.py
```
- This will train and save new model weights and scaler.

## Evaluating Models (Optional)
To evaluate model performance:
```bash
python evaluate_model.py
```

## Notes
- The app fetches the last 120 days of stock data using yfinance.
- Make sure you have a stable internet connection for data download.
- The prediction is for the next trading day (skips weekends).

## File Descriptions
- `app.py`: Main Streamlit app for prediction
- `train_models.py`: Script to train models
- `evaluate_model.py`: Script to evaluate models
- `lstm_model_v2.pth`, `gru_model.pth`, `tcn_model.pth`: Pre-trained model weights
- `scaler_v2.pkl`: Scaler for feature normalization
- `nvidia.csv`: Example CSV data (optional)

---

**Author:** MJH
**Date:** April 2026
