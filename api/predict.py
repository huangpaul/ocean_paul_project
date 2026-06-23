import os
import torch
import torch.nn as nn
import numpy as np

# ---------------------------
# 模型架構定義 (與 baseline_patchTST.py 相同)
# ----------------------------------------
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



class PatchTST_CloserToOfficial(nn.Module):
    def __init__(
        self,
        num_vars: int,
        seq_len: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 1,
        dropout: float = 0.1,
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
        x = self.revin(x, "norm")
        x = x.permute(0, 2, 1).contiguous().view(B * N, L)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = self.patch_proj(x) + self.pos_embed
        x = self.drop(x)
        x = self.encoder(x)
        x = x.reshape(B * N, -1)
        y = self.head(x).view(B, N, self.pred_len)
        y = y.permute(0, 2, 1).contiguous()
        y = self.revin(y, "denorm")
        return y


# ---------------------------
# 推論器類別
# ---------------------------
class PatchTSTPredictor:
    def __init__(self, celsius_model_path: str, humidity_model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 參數需與訓練時一致 (根據 baseline_patchTST.py)
        self.seq_len = 120
        self.pred_len = 60
        self.n_vars = 2
        self.patch_len = 16
        self.stride = 8
        self.d_model = 32
        self.n_heads = 2
        self.n_layers = 2
        self.dropout = 0.3
        
        print(f"Loading models to {self.device}...")
        
        # 初始化 Celsius 模型
        self.model_celsius = self._build_model()
        if os.path.exists(celsius_model_path):
            self.model_celsius.load_state_dict(torch.load(celsius_model_path, map_location=self.device))
        else:
            print(f"Warning: Celsius model not found at {celsius_model_path}")
        self.model_celsius.eval()
        
        # 初始化 Humidity 模型
        self.model_humidity = self._build_model()
        if os.path.exists(humidity_model_path):
            self.model_humidity.load_state_dict(torch.load(humidity_model_path, map_location=self.device))
        else:
            print(f"Warning: Humidity model not found at {humidity_model_path}")
        self.model_humidity.eval()

    def _build_model(self):
        model = PatchTST_CloserToOfficial(
            num_vars=self.n_vars,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)
        return model

    def predict(self, history_data: np.ndarray):
        """
        history_data: numpy array of shape (120, 2), columns: [celsius, humidity]
        returns: dictionary with 'celsius' and 'humidity' predictions of length 60
        """
        if history_data.shape != (self.seq_len, self.n_vars):
            raise ValueError(f"Input shape must be ({self.seq_len}, {self.n_vars})")

        # 轉換為 Tensor: [1, 120, 2]
        x_tensor = torch.FloatTensor(history_data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 預測 Celsius (假設 target_idx = 0)
            pred_c_all = self.model_celsius(x_tensor)
            pred_celsius = pred_c_all[0, :, 0].cpu().numpy() # shape: (60,)
            
            # 預測 Humidity (假設 target_idx = 1)
            pred_h_all = self.model_humidity(x_tensor)
            pred_humidity = pred_h_all[0, :, 1].cpu().numpy() # shape: (60,)
            
        return {
            "celsius": pred_celsius.tolist(),
            "humidity": pred_humidity.tolist()
        }

# 簡單的測試執行區塊
if __name__ == "__main__":
    predictor = PatchTSTPredictor(
        celsius_model_path="/home/user/PatchTST_single/results/20260514_135730/ocean_research/celsius/pred60hr/best_patchtst_model.pth",
        humidity_model_path="/home/user/PatchTST_single/results/20260514_135730/ocean_research/humidity/pred60hr/best_patchtst_model.pth"
    )
    # 建立假資料測試: 120 筆 (溫度, 濕度)
    dummy_data = np.random.rand(120, 2).astype(np.float32)
    res = predictor.predict(dummy_data)
    print("Celsius Predict (first 5):", res['celsius'][:5])
    print("Humidity Predict (first 5):", res['humidity'][:5])
