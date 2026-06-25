#!/bin/bash

# 1. 取得當前日期 (格式: 20260625)
TODAY=$(date +%Y%m%d)

# 2. 載入 Conda 環境 (請確認指向你的 miniconda 實際路徑)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ocean_wsl

# 3. 切換到專案根目錄
cd /home/user/ocean_paul_project

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] 開始自動定期訓練 PatchTST 模型 ==="

# 4. 執行訓練
python train/baseline_patchTST.py

# 5. 💡 動態抓取今天（${TODAY}）最新產出的時間戳記資料夾
# 例如抓到 results/20260625_103519/
LATEST_DIR=$(ls -td results/${TODAY}_*/ 2>/dev/null | head -n 1)

if [ -z "$LATEST_DIR" ]; then
    echo "⚠️ 錯誤：找不到今天 ${TODAY} 產出的結果資料夾！"
    exit 1
fi

echo "🔍 偵測到最新結果目錄: ${LATEST_DIR}"

# 6. 定義模型權重來源（請根據你 results 資料夾下 .pth 檔的實際檔名修改，例如最佳權重可能叫 best.pth 或 model.pth）
SRC_CELSIUS="${LATEST_DIR}celsius/best.pth"
SRC_HUMIDITY="${LATEST_DIR}humidity/best.pth"

# 7. 移動與備份溫度模型
if [ -f "$SRC_CELSIUS" ]; then
    cp "$SRC_CELSIUS" "models/celsius_${TODAY}.pth"
    mv "$SRC_CELSIUS" "models/latest_celsius.pth"
    echo "🎉 溫度模型備份成功: celsius_${TODAY}.pth & latest_celsius.pth"
else
    echo "⚠️ 找不到溫度模型檔案: $SRC_CELSIUS"
fi

# 8. 移動與備份濕度模型
if [ -f "$SRC_HUMIDITY" ]; then
    cp "$SRC_HUMIDITY" "models/humidity_${TODAY}.pth"
    mv "$SRC_HUMIDITY" "models/latest_humidity.pth"
    echo "🎉 濕度模型備份成功: humidity_${TODAY}.pth & latest_humidity.pth"
else
    echo "⚠️ 找不到濕度模型檔案: $SRC_HUMIDITY"
fi


