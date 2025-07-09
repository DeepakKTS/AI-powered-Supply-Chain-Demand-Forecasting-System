import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.model import LSTMForecast  # Ensure this file and class exist


def load_model_and_predict(X):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Ensure input is 3D for LSTM: [batch_size, sequence_length, input_size]
    if X.ndim == 2:
        X = X[:, np.newaxis, :]  # Reshape to [batch, 1, input_size]

    input_size = X.shape[2]  # feature count
    model = LSTMForecast(input_size=input_size)
    model.load_state_dict(torch.load("model/lstm_model.pt", map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        preds = model(X_tensor).cpu().numpy().flatten()

    # Use first feature (sales_qty) as ground truth
    y_true = X[:, -1, 0]  # sales_qty from last time step
    y_pred = preds

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true[y_true != 0] - y_pred[y_true != 0]) / y_true[y_true != 0])) * 100

    return y_pred, {"mae": mae, "rmse": rmse, "mape": mape}, model, X_tensor


def estimate_feature_importance(model, X, y_true):
    """
    Estimate feature importance using permutation method.
    Assumes X is [batch, sequence, features]
    """
    importances = []
    baseline_preds = model(torch.tensor(X, dtype=torch.float32)).detach().cpu().numpy().flatten()
    baseline_error = mean_absolute_error(y_true, baseline_preds)

    for i in range(X.shape[2]):  # for each feature
        X_perm = X.copy()
        np.random.shuffle(X_perm[:, :, i])  # shuffle only the ith feature
        perm_preds = model(torch.tensor(X_perm, dtype=torch.float32)).detach().cpu().numpy().flatten()
        perm_error = mean_absolute_error(y_true, perm_preds)
        importance = perm_error - baseline_error
        importances.append(importance)

    return importances

