# Bicep Generator - LangGraph + FastAPI + Next.js

Azure OpenAIを使ってBicepコードを生成するチャットアプリケーションです。

## 構成

- **`main.py`**: オリジナルのコンソールベースLangGraphアプリ
- **`backend/`**: FastAPI ウェブサーバー（main.pyをAPI化）
- **`frontend/`**: Next.js チャットUI（旧: vscode-chat-editor）

## 前提条件

### 必要なソフトウェア
- Python 3.8+
- Node.js 18+
- pnpm

### Azure OpenAI設定
`.env`ファイルをルートディレクトリに作成し、以下の環境変数を設定してください：

```env
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_API_KEY=your-api-key
```

## セットアップ・実行手順

### 1. 依存関係のインストール

```powershell
# Pythonパッケージ（バックエンド用）
pip install -r requirements.txt

# Node.jsパッケージ（フロントエンド用）
cd frontend
pnpm install
cd ..
```

### 2. バックエンド起動

```powershell
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### 3. フロントエンド起動（別ターミナル）

```powershell
cd frontend
pnpm dev
```

### 4. アプリケーションアクセス

- **フロントエンド**: http://localhost:3000
- **バックエンドAPI**: http://localhost:8000
- **API健康チェック**: http://localhost:8000/health

## 使用方法

1. ブラウザでhttp://localhost:3000にアクセス
2. チャット欄で要件を入力（例：「Webアプリ用のストレージアカウントが欲しい」）
3. AIが質問を重ねて要件を明確化
4. 要件が十分集まったらBicepコードが右側のエディタに表示

## トラブルシューティング

### Windowsでの注意点
- バックエンドのホストは`127.0.0.1`を使用（`0.0.0.0`だとWindowsファイアウォールでブロックされる場合があります）
- ポート8000が使用中の場合は別のポートを指定：`--port 8001`

### デバッグモード
詳細なログを確認したい場合：

```powershell
# デバッグログ有効化
$env:DEBUG_LOG = "1"
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### API直接テスト

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
- `GET /config`: Azure設定確認（APIキー除く）

## ファイル構成

```
TestLangGraph/
├── main.py                 # オリジナルコンソールアプリ
├── requirements.txt        # Python依存関係
├── .env                   # Azure OpenAI設定（要作成）
├── README.md              # このファイル
├── backend/
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