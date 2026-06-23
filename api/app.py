from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import mysql.connector
import pandas as pd
import os

# 載入預測器架構
from predict import PatchTSTPredictor
from dotenv import load_dotenv

# 💡 自動尋找專案根目錄底下的 .env 檔案並載入
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

app = FastAPI(title="PatchTST 溫濕度預測 API (WSL 運算後端)")

# 設定 CORS 允許跨域請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================================================================
# 🛡️ 專案路徑自動化定位 (適用於 ocean_paul_project 結構)
# ===================================================================
# 明確定義出「目前 api/ 資料夾」的位置
CURRENT_API_DIR = os.path.dirname(os.path.abspath(__file__))

# 明確定義出「專案的主根目錄 (ocean_paul_project)」
PROJECT_ROOT = os.path.dirname(CURRENT_API_DIR)

# 往後找任何資料夾，都一律從 PROJECT_ROOT 出發，清晰不數錯層級
CELSIUS_PATH = os.path.join(PROJECT_ROOT, "models", "latest_celsius.pth")
HUMIDITY_PATH = os.path.join(PROJECT_ROOT, "models", "latest_humidity.pth")

# 初始化 Predictor 實例，將設定好的路徑傳入
predictor = PatchTSTPredictor(
    celsius_model_path=CELSIUS_PATH,
    humidity_model_path=HUMIDITY_PATH
)


DB_CONFIG = {
        'host': os.getenv('DB_HOST'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'), 
        'database': os.getenv('DB_NAME'),
        'auth_plugin': 'caching_sha256_password'  # 💡 學習點：改成現代 MySQL 預設的快取 SHA256 驗證
    }
# ===================================================================
# 🚀 核心預測 API 路由
# ===================================================================
# 💡 修正點：改為標準 def，讓 FastAPI 自動使用獨立執行緒池 (Thread Pool) 處理同步資料庫與顯卡運算
@app.get("/api/predict")
def get_prediction():
    """
    從 WSL 本地資料庫讀取過去 120 筆溫濕度資料，並用 5070 顯卡回傳未來 60 筆預測結果
    """
    try:
        # 1. 建立資料庫連線並抓取最新 120 筆資料
        conn = mysql.connector.connect(**DB_CONFIG)
        # 利用主鍵 id 進行 DESC 倒序排序，查詢效率最高
        query = "SELECT reading_time, celsius, humidity FROM temp_readings ORDER BY id DESC LIMIT 120"
        df = pd.read_sql_query(query, conn)
        conn.close()

        if len(df) < 120:
            raise HTTPException(status_code=400, detail=f"WSL 資料庫資料不足 120 筆 (目前只有 {len(df)} 筆)")

        # 2. 將 DESC 的最新資料反轉回正常時間軸順序 (舊 -> 新)
        df = df.iloc[::-1].reset_index(drop=True)
        
        # 3. 處理前端折線圖需要的時間字串標籤
        history_time = pd.to_datetime(df['reading_time']).dt.strftime('%H:%M:%S').tolist()
        
        # 4. 防禦性特徵清洗：在 DataFrame 內直接進行空值填充，避免顯卡吃到 NaN 噴錯
        df['celsius'] = df['celsius'].ffill().bfill().astype(np.float32)
        df['humidity'] = df['humidity'].ffill().bfill().astype(np.float32)

        # 5. 高效記憶體提取：直接利用 .values 拔出形狀為 (120, 2) 的 NumPy 矩陣，取代 column_stack
        history_data = df[['celsius', 'humidity']].values
        
        # 6. 進行 5070 顯卡 PatchTST 模型推論
        result = predictor.predict(history_data)
        
        # 7. 回傳標準格式資料給 151 伺服器代理端
        return {
            "success": True,
            "history": {
                "time": history_time,
                "celsius": df['celsius'].astype(float).tolist(),
                "humidity": df['humidity'].astype(float).tolist()
            },
            "predictions": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "歡迎使用 PatchTST 預測核心！本主機 (WSL/5070) 目前正透過 Tailscale 待命提供運算服務。"}

if __name__ == "__main__":
    import uvicorn
    # 使用 0.0.0.0 監聽，讓來自 Tailscale 內網的流量可以順利連入 WSL
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)