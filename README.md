# Bicep Generator - LangGraph + FastAPI + Next.js

Azure OpenAI を使って Bicep コードを生成するチャットアプリケーションです。

## 構成

- **`main.py`**: オリジナルのコンソールベース LangGraph アプリ
- **`backend/`**: FastAPI ウェブサーバー（main.py を API 化）
- **`frontend/`**: Next.js チャット UI（旧: vscode-chat-editor）

## 前提条件

### 必要なソフトウェア

- Python 3.12+
- Node.js 18+
- pnpm

### Azure OpenAI 設定

`.env`ファイルをルートディレクトリに作成し、以下の環境変数を設定してください：

```env
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_API_KEY=your-api-key
```

## セットアップ・実行手順

### 1. バックエンド起動

```bash
cd backend

# Option 1: venv
python3 -m venv .venv
.venv/bin/activate # Linux/macOS
.venv\Scripts\activate.ps1 # Windows PowerShell
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload

# Option 2: uv
uv run -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### 2. フロントエンド起動（別ターミナル）

```bash
cd frontend
pnpm install
pnpm dev
```

### 3. アプリケーションアクセス

- **フロントエンド**: http://localhost:3000
- **バックエンド API**: http://localhost:8000
- **API 健康チェック**: http://localhost:8000/health

## 使用方法

1. ブラウザで http://localhost:3000 にアクセス
2. チャット欄で要件を入力（例：「Web アプリ用のストレージアカウントが欲しい」）
3. AI が質問を重ねて要件を明確化
4. 要件が十分集まったら Bicep コードが右側のエディタに表示

## トラブルシューティング

### Windows での注意点

- バックエンドのホストは`127.0.0.1`を使用（`0.0.0.0`だと Windows ファイアウォールでブロックされる場合があります）
- ポート 8000 が使用中の場合は別のポートを指定：`--port 8001`
- ポート 3000 が使用中の場合は別のポート (e.g. 3001) でフロントエンドが立ち上がります。この場合、バックエンドへの API コールが CORS エラーになることがあります。`backend/app.py`の CORS 設定を修正してください。

### デバッグモード

詳細なログを確認したい場合：

```powershell
# デバッグログ有効化
$env:DEBUG_LOG = "1"
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### API 直接テスト

```powershell
# 健康チェック
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method GET

# チャットテスト
$body = @{ message = "テストメッセージ" } | ConvertTo-Json
$headers = @{ "Content-Type" = "application/json" }
Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" -Method POST -Body $body -Headers $headers

# セッションリセット
Invoke-RestMethod -Uri "http://127.0.0.1:8000/reset" -Method POST

# Azure設定確認
Invoke-RestMethod -Uri "http://127.0.0.1:8000/config" -Method GET
```

## API エンドポイント

- `GET /`: ウェルカムメッセージ
- `POST /chat`: チャットメッセージ送信
- `POST /reset`: セッションリセット
- `GET /health`: サービス健康状態
- `GET /config`: Azure 設定確認（API キー除く）

## ファイル構成

```
TestLangGraph/
├── .env                   # Azure OpenAI設定（要作成）
├── README.md              # このファイル
├── backend/
│   ├── requirements.txt   # Python依存関係
│   ├── pyproject.toml     # uv用設定
│   └── app.py             # FastAPI サーバー
└── frontend/              # Next.js アプリ
    ├── package.json
    ├── app/
    │   ├── layout.tsx
    │   └── page.tsx       # メインチャットUI
    └── components/        # UIコンポーネント
```

## ライセンス

このプロジェクトはテスト・学習目的で作成されています。
