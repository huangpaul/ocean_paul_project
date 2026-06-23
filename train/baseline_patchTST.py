# input
# RevIN Normalization
# Channel Independence
# Patching
# Patch Embedding
# Transformer Encoder 1
# Linear Prediction Head (線性頭) (單層 nn.Linear 映射至 24 小時)
# RevIN Denormalization
# Output
import os
from dotenv import load_dotenv

# 💡 自動尋找 train/ 資料夾上一層（專案根目錄）底下的 .env 檔案並載入
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

import math
import random
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import mysql.connector

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm

# ---------------------------
# 1) Reproducibility helpers
# ---------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------
# 2) RevIN (clean version)
# ---------------------------
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

        self.mean = None
        self.stdev = None

    def forward(self, x, mode: str):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = x.var(dim=1, keepdim=True, unbiased=False).add(self.eps).sqrt().detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x

        if mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.stdev + self.mean
            return x

        raise ValueError("mode must be 'norm' or 'denorm'")

# ---------------------------
# 3) PatchTST (工業級維度修正版)
# ---------------------------
class PatchTST_CloserToOfficial(nn.Module):
    def __init__(
        self,
        num_vars: int,
        seq_len: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride

        self.revin = RevIN(num_vars, affine=True)
        self.num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patch, d_model))
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model * self.num_patch, pred_len)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, L, N = x.shape
        if N != self.num_vars or L != self.seq_len:
            raise ValueError(f"Shape mismatch: expected ({self.seq_len}, {self.num_vars})")

        x = self.revin(x, "norm")
        x = x.permute(0, 2, 1).contiguous().view(B * N, L)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        x = self.patch_proj(x) + self.pos_embed
        x = self.drop(x)
        x = self.encoder(x)

        # 💡 使用安全平坦化，修正原本的維度 Bug
        x = x.flatten(start_dim=1)
        
        y = self.head(x).view(B, N, self.pred_len)
        y = y.permute(0, 2, 1).contiguous()

        y = self.revin(y, "denorm")
        return y

# ---------------------------
# 4) Dataset
# ---------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, data_np: np.ndarray, seq_len: int, pred_len: int, target_idx: int):
        super().__init__()
        self.data = torch.FloatTensor(data_np.copy().astype(np.float32))
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.target_idx = int(target_idx)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_len, :]
        y = self.data[
            index + self.seq_len:index + self.seq_len + self.pred_len,
            self.target_idx:self.target_idx + 1
        ]
        return x, y

# ---------------------------
# 5) Utilities: metrics + plot
# ---------------------------
def compute_per_horizon_metrics(trues: np.ndarray, preds: np.ndarray):
    pred_len = trues.shape[1]
    rows = []
    for i in range(pred_len):
        mae = mean_absolute_error(trues[:, i], preds[:, i])
        rmse = np.sqrt(mean_squared_error(trues[:, i], preds[:, i]))
        r2 = r2_score(trues[:, i], preds[:, i])
        rows.append([i + 1, mae, rmse, r2])
    return pd.DataFrame(rows, columns=["Hour", "MAE", "RMSE", "R2"])

def plot_metrics(df_metrics: pd.DataFrame, save_path: str, title: str):
    plt.figure(figsize=(10, 5))
    plt.grid(True, which="both", linestyle="-", linewidth=0.5)
    plt.plot(df_metrics["Hour"], df_metrics["RMSE"], marker="o", label="RMSE")
    plt.plot(df_metrics["Hour"], df_metrics["MAE"], marker="x", label="MAE")
    plt.plot(df_metrics["Hour"], df_metrics["R2"], marker="s", label="$R^2$")
    plt.title(title)
    plt.xlabel("Forecast horizon (hour)")
    plt.xticks(range(1, int(df_metrics["Hour"].max()) + 1))
    plt.legend(loc="center right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ---------------------------
# 6) Main
# ---------------------------
if __name__ == "__main__":
    seed_everything(42)

    # ===================================================================
    # 🔐 修正點：全權交給外部 .env 決定，程式碼內不留任何明文密碼字元！
    # ===================================================================
    DB_CONFIG = {
        'host': os.getenv('DB_HOST'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'), 
        'database': os.getenv('DB_NAME'),
        'auth_plugin': 'mysql_native_password'
    }
    
    TABLE_NAME = "temp_readings"

    # ==========================================
    # 🎯 自動動態取得新專案內的路徑
    # ==========================================
    TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(TRAIN_DIR)

    # 精確對齊 ocean_paul_project/models/
    OCEAN_MODELS_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(OCEAN_MODELS_DIR, exist_ok=True)

    # 精確對齊 ocean_paul_project/results/年月日_時分秒/
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    BASE_RESULT_DIR = os.path.join(BASE_DIR, "results", TIMESTAMP)

    TARGET_COLS = ["celsius", "humidity"]

    SEQ_LEN = 120
    PRED_LEN = 60
    BATCH_SIZE = 16
    EPOCHS = 100
    LR = 5e-4
    PATCH_LEN = 16   
    STRIDE = 8      

    D_MODEL = 32
    N_HEADS = 4
    N_LAYERS = 2
    DROPOUT = 0.3

    patience = 10            
    min_delta = 1e-4         
    CLIP_NORM = 1.0

    db_name = "ocean_research"
    target_cols = TARGET_COLS if isinstance(TARGET_COLS, list) else [TARGET_COLS]
    latest_model_names = {
        "celsius": "latest_celsius.pth",
        "humidity": "latest_humidity.pth",
    }

    # 讀取 MySQL 資料
    conn = mysql.connector.connect(**DB_CONFIG)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} ORDER BY reading_time ASC", conn)
    conn.close()

    reading_times = df['reading_time'].values
    numeric_cols = ['celsius', 'humidity']
    data_df = df[numeric_cols].copy().astype('float32')
    n_vars = data_df.shape[1]

    # 切分時序資料集
    n = len(data_df)
    train_idx_end = int(n * 0.70)
    val_idx_end = int(n * 0.85)
    
    train_np = data_df.iloc[0:train_idx_end].values
    val_np = data_df.iloc[train_idx_end:val_idx_end].values
    test_np = data_df.iloc[val_idx_end:].values

    print("=" * 80)
    print("DMC LAB AUTOMATED TRAINING INITIALIZED")
    print("=" * 80)
    print(f"Total records: {n} | Save Base: results/{TIMESTAMP}/")
    print("=" * 80)

    for TARGET_COL in target_cols:
        if TARGET_COL not in data_df.columns:
            raise ValueError(f"TARGET_COL '{TARGET_COL}' not found.")

        target_idx = data_df.columns.get_loc(TARGET_COL)
        
        # 💡 修正路徑：讓歷史報告完美存入新結果資料夾
        RESULT_DIR = os.path.join(BASE_RESULT_DIR, TARGET_COL)
        os.makedirs(RESULT_DIR, exist_ok=True)

        train_ds = TimeSeriesDataset(train_np, SEQ_LEN, PRED_LEN, target_idx=target_idx)
        val_ds = TimeSeriesDataset(val_np, SEQ_LEN, PRED_LEN, target_idx=target_idx)
        test_ds = TimeSeriesDataset(test_np, SEQ_LEN, PRED_LEN, target_idx=target_idx)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PatchTST_CloserToOfficial(
            num_vars=n_vars, seq_len=SEQ_LEN, pred_len=PRED_LEN,
            patch_len=PATCH_LEN, stride=STRIDE, d_model=D_MODEL,
            n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
        criterion = nn.MSELoss()

        best_val = float("inf")
        best_path = os.path.join(RESULT_DIR, "best_patchtst_model.pth")
        wait = 0

        print(f"\n>> 正在訓練特徵目標: {TARGET_COL}...")
        for epoch in range(EPOCHS):
            model.train()
            running = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", ncols=110)
            for bx, by in pbar:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred_all = model(bx)
                pred_t = pred_all[:, :, target_idx:target_idx + 1]
                loss = criterion(pred_t, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                optimizer.step()
                running += loss.item()
                pbar.set_postfix(train_loss=f"{loss.item():.5f}")

            model.eval()
            vloss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    pred_all = model(bx)
                    pred_t = pred_all[:, :, target_idx:target_idx + 1]
                    vloss += criterion(pred_t, by).item()

            avg_train = running / len(train_loader)
            avg_val = vloss / len(val_loader)
            scheduler.step(avg_val)

            print(f"   [Epoch {epoch + 1}] Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

            if (best_val - avg_val) > min_delta:
                best_val = avg_val
                wait = 0
                torch.save(model.state_dict(), best_path)
                print(f"   🔥 刷新最佳紀錄 -> 權重已存至 Snapshot")
            else:
                wait += 1
                if wait >= patience:
                    print(f"   🛑 早停機制觸發，進入測試評估。")
                    break

        # 測試集效能測試
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                pred_all = model(bx)
                all_preds.append(pred_all[:, :, target_idx].cpu().numpy())
                all_trues.append(by[:, :, 0].cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)

        np.save(os.path.join(RESULT_DIR, "preds.npy"), preds)
        np.save(os.path.join(RESULT_DIR, "trues.npy"), trues)

        hourly_df = compute_per_horizon_metrics(trues, preds)
        hourly_df.to_csv(os.path.join(RESULT_DIR, "hourly_metrics_24h.csv"), index=False)
        plot_metrics(hourly_df, os.path.join(RESULT_DIR, "test_metrics_per_horizon.png"), f"Test metrics - {TARGET_COL}")

        # 🚀 自動同步：把訓練好的最佳模型複製到目前的 ocean_paul_project/models/latest_*.pth
        if TARGET_COL in latest_model_names:
            dst_production_path = os.path.join(OCEAN_MODELS_DIR, latest_model_names[TARGET_COL])
            shutil.copy2(best_path, dst_production_path)
            print(f"🚀 最新最佳權重已自動同步至生產線 -> {dst_production_path}")

    print(f"\n全部訓練任務已完成！報告已儲存至 results/{TIMESTAMP}/")