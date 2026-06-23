# 🌊 Ocean Paul Project - PatchTST 溫濕度預測微服務系統

本專案是一個基於 **PatchTST (深度時序時間序列模型)** 的雙特徵（溫度、濕度）預測微服務系統。系統採用動靜分離架構，將 AI 模型研發訓練（Train）與線上即時推論服務（FastAPI）徹底解耦，並實作 100% 密碼抽離的工業級安全防禦機制。

---

## 📂 專案目錄架構

```text
ocean_paul_project/
├── .env                # 🔐 環境變數設定檔 (集中管理資料庫密碼，本地獨有，不上傳 Git)
├── .gitignore          # 🚫 Git 忽略清單 (自動過濾大檔案與敏感資訊)
├── requirements.txt    # 📦 專案依賴套件清單
├── README.md           # 📖 專案說明文件
├── api/                # 🚀 生產線 API 推論微服務
│   ├── app.py          # FastAPI 主程式 (Port 8000 服務接線生)
│   └── predict.py      # PatchTST 顯卡推論核心
├── models/             # 💾 最新生產線權重儲存區 (*.pth)
└── train/              # 🏋️ AI 模型研發與訓練主程式
    └── baseline_patchTST.py