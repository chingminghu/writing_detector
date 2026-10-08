# 鍵盤軌跡數字辨識

以 PyTorch GRU 模型根據鍵盤座標序列預測數字 0–9，並透過 FastAPI 提供網頁與 JSON API。

## 本機執行

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload
```

開啟 <http://127.0.0.1:8000>。API 文件位於 <http://127.0.0.1:8000/docs>。

## API

```http
POST /api/predict
Content-Type: application/json

{"text":"qazxswedc"}
```

健康檢查：`GET /health`

