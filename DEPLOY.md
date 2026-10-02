# 部署到 Streamlit Community Cloud

1. 將這個資料夾上傳到 GitHub repository。
2. 確認 repository 根目錄包含 `app.py`、`requirements.txt`、`literature_db.xlsx`。
3. 在 Streamlit Community Cloud 建立新 App。
4. 選擇這個 repository，Main file path 填入 `app.py`。
5. 開啟 App settings → Secrets，依 `.streamlit/secrets.example.toml` 填入：
   - Microsoft Foundry 模型推論 endpoint
   - 模型 deployment 名稱
   - API key
6. 部署完成後，先用電腦與手機各測一次：
   - 操作與結果
   - Microsoft 研究助理能回答問題並顯示文獻編號
   - 文獻資料
   - 目前模型差多少
   - 這個工具怎麼開始的
7. 確認公開網址後，再把網址轉成 QR Code 放進競賽資料。

## 公開前最後檢查

- 不要放 `.env`、`secrets.toml` 或私人資料。
- 不要把 Microsoft Foundry API key 寫進程式、截圖或 GitHub commit。
- `literature_db.xlsx` 會跟著公開 repository；如果裡面有不想公開的欄位，先刪除。
- 這個工具目前定位為研究整理與條件探索原型，不作為實際製程 recipe 的依據。
