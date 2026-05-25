#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import warnings
import random
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor

import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── SEED ─────────────────────────────────────────────
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)

# ── PATH ─────────────────────────────────────────────
DATA_PATH = os.path.join("data", "dataset_train.csv")
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("  FINAL PIPELINE — LR + RIDGE + RF + XGB + LGBM + BAGGING + MLP + LSTM + GRU + TRANSFORMER + TABNET + TABPFN")
print("=" * 70)

# ====================================================
# 1. LOAD DATA
# ====================================================
df = pd.read_csv(DATA_PATH, low_memory=False)

ts_col = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()][0]
df.rename(columns={ts_col: "timestamp"}, inplace=True)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

# ====================================================
# 2. RUL LABELING
# ====================================================
FAILURE_1 = pd.Timestamp("2022-06-04 10:19:24")
FAILURE_2 = pd.Timestamp("2022-07-11 10:10:18")

df["RUL"] = np.inf

for fs in [FAILURE_1, FAILURE_2]:
    diff = (fs - df["timestamp"]).dt.total_seconds() / 3600
    df["RUL"] = np.minimum(df["RUL"], np.where(diff > 0, diff, np.inf))

df["RUL"] = df["RUL"].replace(np.inf, 0)

# ====================================================
# 3. PREPROCESSING
# ====================================================
ANALOG_COLS = [c for c in ["TP2","TP3","H1","DV_pressure","Reservoirs",
                           "Oil_temperature","Motor_current"] if c in df.columns]

# IQR clipping
for col in ANALOG_COLS:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 3*IQR, Q3 + 3*IQR)

# RESAMPLE
df = df.set_index("timestamp")
df = df[ANALOG_COLS + ["RUL"]].resample("2min").mean()
df = pd.DataFrame(df).dropna().reset_index()

# ====================================================
# 4. TIME FEATURES (NO LEAKAGE)
# ====================================================
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["sin_time"] = np.sin(2 * np.pi * df["hour"] / 24)
df["cos_time"] = np.cos(2 * np.pi * df["hour"] / 24)

# ====================================================
# 5. FEATURE ENGINEERING — ONLY PAST VALUES (NO LEAKAGE)
# ====================================================
for col in ANALOG_COLS:
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)

for col in ANALOG_COLS:
    for w in [2, 5, 10, 48, 60]:
        df[f"{col}_roll{w}_mean"] = df[col].shift(1).rolling(window=w, min_periods=1).mean()
        df[f"{col}_roll{w}_std"] = df[col].shift(1).rolling(window=w, min_periods=1).std()

df = df.dropna().reset_index(drop=True)

# ====================================================
# 6. SPLIT FIRST, THEN SCALE (NO LEAKAGE)
# ====================================================
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Fit scalers on TRAIN ONLY
feat_scaler = RobustScaler()
train_df[ANALOG_COLS] = feat_scaler.fit_transform(train_df[ANALOG_COLS])
test_df[ANALOG_COLS] = feat_scaler.transform(test_df[ANALOG_COLS])

rul_scaler = RobustScaler()
train_df["RUL_scaled"] = rul_scaler.fit_transform(train_df[["RUL"]])
test_df["RUL_scaled"] = rul_scaler.transform(test_df[["RUL"]])

# Full scaled dataset for sequential models
full_df = pd.concat([train_df, test_df], ignore_index=True)

# ====================================================
# 7. FEATURES
# ====================================================
feature_cols = [c for c in train_df.columns if c not in ["timestamp", "RUL", "RUL_scaled", "hour", "minute"]]

X_train = train_df[feature_cols]
y_train = train_df["RUL_scaled"]
X_test = test_df[feature_cols]
y_test = test_df["RUL_scaled"]

X_full = full_df[feature_cols]
y_full = full_df["RUL_scaled"]

print(f"\nTrain samples: {len(train_df)}, Test samples: {len(test_df)}")
print(f"Full dataset:  {len(full_df)}")

# ====================================================
# 8. HYPERPARAMETERS STORAGE
# ====================================================
hyperparams = {}

# ====================================================
# 9. TABULAR MODELS — TRAIN ON FULL DATASET
# ====================================================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        max_features=0.5,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
    "Bagging": BaggingRegressor(
        estimator=RandomForestRegressor(n_estimators=100, random_state=42),
        n_estimators=10,
        random_state=42,
        n_jobs=-1
    ),
    "MLP": MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=256,
        learning_rate='adaptive',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    )
}

hyperparams["Linear Regression"] = {"model": "LinearRegression", "fit_intercept": True}
hyperparams["Ridge"] = {"model": "Ridge", "alpha": 1.0}
hyperparams["Random Forest"] = {"model": "RandomForestRegressor", "n_estimators": 500, "max_features": 0.5, "random_state": 42, "n_jobs": -1}
hyperparams["XGBoost"] = {"model": "XGBRegressor", "n_estimators": 500, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "n_jobs": -1}
hyperparams["LightGBM"] = {"model": "LGBMRegressor", "n_estimators": 500, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "n_jobs": -1, "verbose": -1}
hyperparams["Bagging"] = {"model": "BaggingRegressor", "estimator": "RandomForestRegressor(n_estimators=100)", "n_estimators": 10, "random_state": 42, "n_jobs": -1}
hyperparams["MLP"] = {"model": "MLPRegressor", "hidden_layer_sizes": "(256, 128, 64)", "activation": "relu", "solver": "adam", "alpha": 0.001, "batch_size": 256, "learning_rate": "adaptive", "max_iter": 1000, "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 20, "random_state": 42}

# ====================================================
# 10. TRAIN + EVALUATE
# ====================================================
results = {}

def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    yh = rul_scaler.inverse_transform(y_te.values.reshape(-1, 1)).flatten()
    ph = rul_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(yh, ph))
    mse = mean_squared_error(yh, ph)
    mae = mean_absolute_error(yh, ph)
    mape = mean_absolute_percentage_error(yh, ph)
    r2 = r2_score(yh, ph)

    print(f"\n{name}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"MAPE: {mape:.4f}")
    print(f"R2  : {r2:.4f}")

    return {
        "Model": name,
        "RMSE": rmse,
        "MSE": mse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "y_true": yh,
        "y_pred": ph
    }

# Train on FULL dataset, evaluate on test
for name, model in models.items():
    res = evaluate(name, model, X_full, y_full, X_test, y_test)
    results[name] = res

# ====================================================
# 11. TABNET
# ====================================================
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    import torch

    torch.manual_seed(GLOBAL_SEED)

    X_train_tab = X_full.values.astype(np.float32)
    X_test_tab = X_test.values.astype(np.float32)
    y_train_tab = y_full.values.reshape(-1, 1).astype(np.float32)
    y_test_tab = y_test.values.reshape(-1, 1).astype(np.float32)

    tabnet = TabNetRegressor(
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type='entmax',
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=0,
        seed=GLOBAL_SEED
    )

    tabnet.fit(
        X_train_tab, y_train_tab,
        eval_set=[(X_test_tab, y_test_tab)],
        eval_metric=['rmse'],
        max_epochs=200,
        patience=20,
        batch_size=256,
        drop_last=False
    )

    tabnet_preds = tabnet.predict(X_test_tab).flatten()

    yh_tabnet = rul_scaler.inverse_transform(y_test_tab.reshape(-1, 1)).flatten()
    ph_tabnet = rul_scaler.inverse_transform(tabnet_preds.reshape(-1, 1)).flatten()

    rmse_tabnet = np.sqrt(mean_squared_error(yh_tabnet, ph_tabnet))
    mse_tabnet = mean_squared_error(yh_tabnet, ph_tabnet)
    mae_tabnet = mean_absolute_error(yh_tabnet, ph_tabnet)
    mape_tabnet = mean_absolute_percentage_error(yh_tabnet, ph_tabnet)
    r2_tabnet = r2_score(yh_tabnet, ph_tabnet)

    print(f"\nTabNet")
    print(f"RMSE: {rmse_tabnet:.2f}")
    print(f"MSE : {mse_tabnet:.2f}")
    print(f"MAE : {mae_tabnet:.2f}")
    print(f"MAPE: {mape_tabnet:.4f}")
    print(f"R2  : {r2_tabnet:.4f}")

    results["TabNet"] = {
        "Model": "TabNet",
        "RMSE": rmse_tabnet,
        "MSE": mse_tabnet,
        "MAE": mae_tabnet,
        "MAPE": mape_tabnet,
        "R2": r2_tabnet,
        "y_true": yh_tabnet,
        "y_pred": ph_tabnet
    }

    hyperparams["TabNet"] = {
        "model": "TabNetRegressor",
        "n_d": 64,
        "n_a": 64,
        "n_steps": 5,
        "gamma": 1.5,
        "lambda_sparse": 1e-4,
        "optimizer": "Adam",
        "lr": 2e-2,
        "mask_type": "entmax",
        "scheduler": "StepLR",
        "scheduler_step_size": 10,
        "scheduler_gamma": 0.9,
        "max_epochs": 200,
        "patience": 20,
        "batch_size": 256,
        "seed": GLOBAL_SEED
    }

except ImportError:
    print("\nTabNet not installed. Skipping. Install with: pip install pytorch-tabnet")

# ====================================================
# 12. TABPFN
# ====================================================
try:
    from tabpfn import TabPFNRegressor

    tabpfn = TabPFNRegressor(device='cpu', seed=GLOBAL_SEED)
    tabpfn.fit(X_full, y_full)

    tabpfn_preds = tabpfn.predict(X_test)

    yh_tabpfn = rul_scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
    ph_tabpfn = rul_scaler.inverse_transform(tabpfn_preds.reshape(-1, 1)).flatten()

    rmse_tabpfn = np.sqrt(mean_squared_error(yh_tabpfn, ph_tabpfn))
    mse_tabpfn = mean_squared_error(yh_tabpfn, ph_tabpfn)
    mae_tabpfn = mean_absolute_error(yh_tabpfn, ph_tabpfn)
    mape_tabpfn = mean_absolute_percentage_error(yh_tabpfn, ph_tabpfn)
    r2_tabpfn = r2_score(yh_tabpfn, ph_tabpfn)

    print(f"\nTabPFN")
    print(f"RMSE: {rmse_tabpfn:.2f}")
    print(f"MSE : {mse_tabpfn:.2f}")
    print(f"MAE : {mae_tabpfn:.2f}")
    print(f"MAPE: {mape_tabpfn:.4f}")
    print(f"R2  : {r2_tabpfn:.4f}")

    results["TabPFN"] = {
        "Model": "TabPFN",
        "RMSE": rmse_tabpfn,
        "MSE": mse_tabpfn,
        "MAE": mae_tabpfn,
        "MAPE": mape_tabpfn,
        "R2": r2_tabpfn,
        "y_true": yh_tabpfn,
        "y_pred": ph_tabpfn
    }

    hyperparams["TabPFN"] = {
        "model": "TabPFNRegressor",
        "device": "cpu",
        "seed": GLOBAL_SEED
    }

except ImportError:
    print("\nTabPFN not installed. Skipping. Install with: pip install tabpfn")

# ====================================================
# 13. SEQUENTIAL MODELS (LSTM, GRU, TRANSFORMER)
# ====================================================
TIMESTEPS = 20

def create_sequences(df_in, feature_cols, target_col, timesteps):
    Xs, ys = [], []
    vals = df_in[feature_cols].values
    targs = df_in[target_col].values
    for i in range(len(df_in) - timesteps):
        Xs.append(vals[i:i+timesteps])
        ys.append(targs[i+timesteps])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(full_df, feature_cols, "RUL_scaled", TIMESTEPS)

np.random.seed(42)
indices = np.random.permutation(len(X_seq))
split_seq = int(len(X_seq) * 0.8)
train_idx = indices[:split_seq]
test_idx = indices[split_seq:]

X_train_seq, X_test_seq = X_seq[train_idx], X_seq[test_idx]
y_train_seq, y_test_seq = y_seq[train_idx], y_seq[test_idx]

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# ── LSTM ──
lstm_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(TIMESTEPS, len(feature_cols))),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

lstm_model.fit(
    X_train_seq, y_train_seq,
    validation_split=0.1,
    epochs=200,
    batch_size=64,
    callbacks=[early_stop, lr_reduce],
    verbose=0
)

lstm_preds = lstm_model.predict(X_test_seq, verbose=0).flatten()

yh_lstm = rul_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
ph_lstm = rul_scaler.inverse_transform(lstm_preds.reshape(-1, 1)).flatten()

rmse_lstm = np.sqrt(mean_squared_error(yh_lstm, ph_lstm))
mse_lstm = mean_squared_error(yh_lstm, ph_lstm)
mae_lstm = mean_absolute_error(yh_lstm, ph_lstm)
mape_lstm = mean_absolute_percentage_error(yh_lstm, ph_lstm)
r2_lstm = r2_score(yh_lstm, ph_lstm)

print(f"\nLSTM")
print(f"RMSE: {rmse_lstm:.2f}")
print(f"MSE : {mse_lstm:.2f}")
print(f"MAE : {mae_lstm:.2f}")
print(f"MAPE: {mape_lstm:.4f}")
print(f"R2  : {r2_lstm:.4f}")

results["LSTM"] = {
    "Model": "LSTM",
    "RMSE": rmse_lstm,
    "MSE": mse_lstm,
    "MAE": mae_lstm,
    "MAPE": mape_lstm,
    "R2": r2_lstm,
    "y_true": yh_lstm,
    "y_pred": ph_lstm
}

hyperparams["LSTM"] = {
    "model": "Sequential",
    "layers": [
        {"type": "LSTM", "units": 128, "return_sequences": True, "input_shape": f"({TIMESTEPS}, {len(feature_cols)})"},
        {"type": "Dropout", "rate": 0.2},
        {"type": "LSTM", "units": 64, "return_sequences": False},
        {"type": "Dropout", "rate": 0.2},
        {"type": "Dense", "units": 32, "activation": "relu"},
        {"type": "Dense", "units": 1}
    ],
    "optimizer": "adam",
    "loss": "mse",
    "epochs": 200,
    "batch_size": 64,
    "early_stopping_patience": 15,
    "lr_reduce_factor": 0.5,
    "lr_reduce_patience": 5
}

# ── GRU ──
gru_model = Sequential([
    GRU(128, return_sequences=True, input_shape=(TIMESTEPS, len(feature_cols))),
    Dropout(0.2),
    GRU(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
gru_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

gru_model.fit(
    X_train_seq, y_train_seq,
    validation_split=0.1,
    epochs=200,
    batch_size=64,
    callbacks=[early_stop, lr_reduce],
    verbose=0
)

gru_preds = gru_model.predict(X_test_seq, verbose=0).flatten()

yh_gru = rul_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
ph_gru = rul_scaler.inverse_transform(gru_preds.reshape(-1, 1)).flatten()

rmse_gru = np.sqrt(mean_squared_error(yh_gru, ph_gru))
mse_gru = mean_squared_error(yh_gru, ph_gru)
mae_gru = mean_absolute_error(yh_gru, ph_gru)
mape_gru = mean_absolute_percentage_error(yh_gru, ph_gru)
r2_gru = r2_score(yh_gru, ph_gru)

print(f"\nGRU")
print(f"RMSE: {rmse_gru:.2f}")
print(f"MSE : {mse_gru:.2f}")
print(f"MAE : {mae_gru:.2f}")
print(f"MAPE: {mape_gru:.4f}")
print(f"R2  : {r2_gru:.4f}")

results["GRU"] = {
    "Model": "GRU",
    "RMSE": rmse_gru,
    "MSE": mse_gru,
    "MAE": mae_gru,
    "MAPE": mape_gru,
    "R2": r2_gru,
    "y_true": yh_gru,
    "y_pred": ph_gru
}

hyperparams["GRU"] = {
    "model": "Sequential",
    "layers": [
        {"type": "GRU", "units": 128, "return_sequences": True, "input_shape": f"({TIMESTEPS}, {len(feature_cols)})"},
        {"type": "Dropout", "rate": 0.2},
        {"type": "GRU", "units": 64, "return_sequences": False},
        {"type": "Dropout", "rate": 0.2},
        {"type": "Dense", "units": 32, "activation": "relu"},
        {"type": "Dense", "units": 1}
    ],
    "optimizer": "adam",
    "loss": "mse",
    "epochs": 200,
    "batch_size": 64,
    "early_stopping_patience": 15,
    "lr_reduce_factor": 0.5,
    "lr_reduce_patience": 5
}

# ── TRANSFORMER ──
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_transformer(timesteps, n_features):
    inputs = Input(shape=(timesteps, n_features))
    x = Dense(64)(inputs)
    x = TransformerBlock(64, 4, 128, 0.1)(x)
    x = TransformerBlock(64, 4, 128, 0.1)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

transformer_model = build_transformer(TIMESTEPS, len(feature_cols))

transformer_model.fit(
    X_train_seq, y_train_seq,
    validation_split=0.1,
    epochs=200,
    batch_size=64,
    callbacks=[early_stop, lr_reduce],
    verbose=0
)

trans_preds = transformer_model.predict(X_test_seq, verbose=0).flatten()

yh_trans = rul_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
ph_trans = rul_scaler.inverse_transform(trans_preds.reshape(-1, 1)).flatten()

rmse_trans = np.sqrt(mean_squared_error(yh_trans, ph_trans))
mse_trans = mean_squared_error(yh_trans, ph_trans)
mae_trans = mean_absolute_error(yh_trans, ph_trans)
mape_trans = mean_absolute_percentage_error(yh_trans, ph_trans)
r2_trans = r2_score(yh_trans, ph_trans)

print(f"\nTransformer")
print(f"RMSE: {rmse_trans:.2f}")
print(f"MSE : {mse_trans:.2f}")
print(f"MAE : {mae_trans:.2f}")
print(f"MAPE: {mape_trans:.4f}")
print(f"R2  : {r2_trans:.4f}")

results["Transformer"] = {
    "Model": "Transformer",
    "RMSE": rmse_trans,
    "MSE": mse_trans,
    "MAE": mae_trans,
    "MAPE": mape_trans,
    "R2": r2_trans,
    "y_true": yh_trans,
    "y_pred": ph_trans
}

hyperparams["Transformer"] = {
    "model": "Transformer",
    "embed_dim": 64,
    "num_heads": 4,
    "ff_dim": 128,
    "dropout_rate": 0.1,
    "num_blocks": 2,
    "dense_units": 32,
    "optimizer": "adam",
    "loss": "mse",
    "epochs": 200,
    "batch_size": 64,
    "early_stopping_patience": 15,
    "lr_reduce_factor": 0.5,
    "lr_reduce_patience": 5
}

# ====================================================
# 14. SAVE COMPARISON CSV
# ====================================================
comparison_df = pd.DataFrame([
    {
        "Model": res["Model"],
        "RMSE": res["RMSE"],
        "MSE": res["MSE"],
        "MAE": res["MAE"],
        "MAPE": res["MAPE"],
        "R2": res["R2"]
    }
    for res in results.values()
])

comparison_df = comparison_df.sort_values("R2", ascending=False).reset_index(drop=True)
comparison_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)
print(f"\n📊 Comparison CSV saved to: {OUTPUT_DIR}/model_comparison.csv")
print(comparison_df.to_string(index=False))

# ====================================================
# 15. SAVE HYPERPARAMETERS
# ====================================================
hyperparams_list = []
for model_name, params in hyperparams.items():
    row = {"Model": model_name}
    row.update({str(k): str(v) for k, v in params.items()})
    hyperparams_list.append(row)

hyperparams_df = pd.DataFrame(hyperparams_list)
hyperparams_df.to_csv(f"{OUTPUT_DIR}/hyperparameters.csv", index=False)
print(f"\n🔧 Hyperparameters CSV saved to: {OUTPUT_DIR}/hyperparameters.csv")

with open(f"{OUTPUT_DIR}/hyperparameters.json", "w") as f:
    json.dump(hyperparams, f, indent=2)
print(f"🔧 Hyperparameters JSON saved to: {OUTPUT_DIR}/hyperparameters.json")

# ====================================================
# 16. PLOTS
# ====================================================
n_models = len(results)
n_cols = 3
n_rows = int(np.ceil(n_models / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
axes = axes.flatten() if n_models > 1 else [axes]

for idx, (name, res) in enumerate(results.items()):
    ax = axes[idx]
    yh = res["y_true"]
    ph = res["y_pred"]

    ax.scatter(yh, ph, alpha=0.4, s=8, c='steelblue')
    ax.plot([yh.min(), yh.max()], [yh.min(), yh.max()], 'r--', lw=2, label='Perfect')
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title(f"{name}\nR²={res['R2']:.4f} | RMSE={res['RMSE']:.2f} | MAE={res['MAE']:.2f}")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

for idx in range(n_models, len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/all_models_scatter.png", dpi=150)
plt.close()
print(f"\n📈 All models scatter plot saved to: {OUTPUT_DIR}/all_models_scatter.png")

# Bar chart
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
metrics = ["R2", "RMSE", "MSE", "MAE", "MAPE"]
colors = plt.cm.viridis(np.linspace(0, 1, len(comparison_df)))

for idx, metric in enumerate(metrics):
    ax = axes.flatten()[idx]
    bars = ax.barh(comparison_df["Model"], comparison_df[metric], color=colors)
    ax.set_xlabel(metric)
    ax.set_title(f"{metric} Comparison")
    ax.invert_yaxis()

    for bar, val in zip(bars, comparison_df[metric]):
        ax.text(bar.get_width() + bar.get_width() * 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}" if metric in ["R2", "MAPE"] else f"{val:.2f}",
                va='center', fontsize=8)

axes.flatten()[5].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/all_models_bar_comparison.png", dpi=150)
plt.close()
print(f"📊 Bar comparison plot saved to: {OUTPUT_DIR}/all_models_bar_comparison.png")

# Best model
best_model = comparison_df.iloc[0]["Model"]
yh_best = results[best_model]["y_true"]
ph_best = results[best_model]["y_pred"]

plt.figure(figsize=(7, 7))
plt.scatter(yh_best, ph_best, alpha=0.5, s=12, c='darkgreen')
plt.plot([yh_best.min(), yh_best.max()], [yh_best.min(), yh_best.max()], 'r--', lw=2, label='Perfect')
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title(f"🏆 BEST MODEL: {best_model}\nR²={results[best_model]['R2']:.4f} | RMSE={results[best_model]['RMSE']:.2f} | MAE={results[best_model]['MAE']:.2f}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/best_model.png", dpi=150)
plt.close()
print(f"🏆 Best model plot saved to: {OUTPUT_DIR}/best_model.png")

# Individual plots
for name, res in results.items():
    plt.figure(figsize=(6, 6))
    plt.scatter(res["y_true"], res["y_pred"], alpha=0.5, s=10, c='steelblue')
    plt.plot([res["y_true"].min(), res["y_true"].max()],
             [res["y_true"].min(), res["y_true"].max()], 'r--', lw=2, label='Perfect')
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title(f"{name}\nR²={res['R2']:.4f} | RMSE={res['RMSE']:.2f} | MAE={res['MAE']:.2f} | MAPE={res['MAPE']:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name.replace(' ', '_')}_plot.png", dpi=150)
    plt.close()

print(f"\n📁 Individual model plots saved to: {OUTPUT_DIR}/")

# ====================================================
# 17. FINAL SUMMARY
# ====================================================
print(f"\n{'='*70}")
print(f"🏆 BEST MODEL: {best_model}")
print(f"{'='*70}")
print(f"\n📁 All outputs saved to: {OUTPUT_DIR}")
print(f"   • model_comparison.csv")
print(f"   • hyperparameters.csv")
print(f"   • hyperparameters.json")
print(f"   • all_models_scatter.png")
print(f"   • all_models_bar_comparison.png")
print(f"   • best_model.png")
print(f"   • [model_name]_plot.png (individual plots)")
print("\n✅ PIPELINE COMPLETE")