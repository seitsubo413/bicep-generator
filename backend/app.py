import os
import re
from typing import List, Dict, Any, Annotated, TypedDict, Optional

import dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages, AnyMessage

# ──────────────────────────────────────────────────────────────────────────────
# Checkpointer: SqliteSaver が無い環境では自動で MemorySaver に切替
# ──────────────────────────────────────────────────────────────────────────────
SqliteSaver = None
MemorySaver = None
try:
    from langgraph.checkpoint.sqlite import SqliteSaver as _SqliteSaver
    SqliteSaver = _SqliteSaver
except Exception:
    try:
        from langgraph.checkpoint.memory import MemorySaver as _MemorySaver
        MemorySaver = _MemorySaver
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# 環境変数
# ──────────────────────────────────────────────────────────────────────────────
dotenv.load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")

DEBUG_LOG = os.getenv("DEBUG_LOG", "0") in ("1", "true", "True")

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI 初期化 & CORS
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Bicep Generator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# LLM クライアント
# ──────────────────────────────────────────────────────────────────────────────
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    api_version=OPENAI_API_VERSION,
)

# ──────────────────────────────────────────────────────────────────────────────
# State / I/O モデル
# ──────────────────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    n_callings: int
    current_user_message: str
    bicep_code: str  # 完了時に格納（判定に使う）

class ChatMessage(BaseModel):
    content: str
    sender: str
    timestamp: Optional[str] = None  # 使わない場合は省略可

class ChatRequest(BaseModel):
    session_id: Optional[str] = "default"  # フロントから会話IDを渡すのが推奨
    message: Optional[str] = None          # 空の場合は「AIの次のステップだけ」進める
    conversation_history: List[ChatMessage] = []  # 未使用（保持はcheckpointerに任せる）

class ChatResponse(BaseModel):
    message: str
    bicep_code: str = ""
    is_complete: bool = False

# ──────────────────────────────────────────────────────────────────────────────
# ノード定義（プロンプトはそのまま）
# ──────────────────────────────────────────────────────────────────────────────
async def hearing(state: State):
    """要件ヒアリング: 1問だけ短く投げる"""
    # LangChainメッセージ → OpenAI形式ヒストリ
    history: List[Dict[str, str]] = []
    for msg in state.get("messages", []):
        if hasattr(msg, "type") and hasattr(msg, "content"):
            if msg.type == "human":
                history.append({"role": "user", "content": msg.content})
            elif msg.type == "ai":
                history.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, dict) and "role" in msg and "content" in msg:
            history.append({"role": str(msg["role"]), "content": str(msg["content"])})
        elif isinstance(msg, str):
            history.append({"role": "assistant", "content": msg})

    messages = [
        {"role": "system", "content": """
             ## 役割
             あなたは優秀な要件ヒアリング担当者です。ユーザーの要件を深掘りして、必要な情報を引き出すための質問を生成するのがあなたの役割です。
             
             ## 目的
             ユーザーは、最終的に Azure 上に環境を構築しようとしています。ヒアリング項目はあくまでも、Azure 上に環境を構築するために必要な情報に限定してください。
             
             ## 考え方
             まだ、背景が明確になっていない場合には、まずは背景を明確にするための質問をしてください。
             """}
    ] + history + [
        {"role": "user", "content": "ユーザーの要件を深掘りするための質問をしてください。ただし、質問は一つだけにしてください。分量は短く、簡潔にしてください。"}
    ]
    resp = await llm.ainvoke(messages)
    question = (resp.content or "").strip() or "要件をもう少し教えてください。"
    return {
        "messages": [{"role": "assistant", "content": question}],
        "n_callings": state.get("n_callings", 0) + 1,
        "current_user_message": state.get("current_user_message", ""),
        "bicep_code": state.get("bicep_code", ""),
    }

def should_hear_again(state: State) -> str:
    """ヒアリング継続判定: 'done' or 'again' を返す（同期関数）"""
    history: List[Dict[str, str]] = []
    for msg in state.get("messages", []):
        if hasattr(msg, "type") and hasattr(msg, "content"):
            if msg.type == "human":
                history.append({"role": "user", "content": msg.content})
            elif msg.type == "ai":
                history.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, dict) and "role" in msg and "content" in msg:
            history.append({"role": str(msg["role"]), "content": str(msg["content"])})
        elif isinstance(msg, str):
            history.append({"role": "assistant", "content": msg})

    messages = [
        {"role": "system", "content": """
あなたは、Azure環境を誰が作っても9割同じになる程度に要件が満ちたか判定します。
満ちていれば 'done'、不足していれば 'again'。回答は 'done' または 'again' のみ。
"""}
    ] + history + [
        {"role": "user", "content": "要件は十分ですか？ 'done' か 'again' で答えてください。"}
    ]

    # guard: ヒアリング上限
    if state.get("n_callings", 0) >= 10:
        return "done"

    # 同期的なLLM呼び出し
    resp = llm.invoke(messages)
    ans = (resp.content or "").strip().lower()
    return "done" if "done" in ans else "again"

async def code_generation(state: State):
    """Bicep コード生成"""
    history: List[Dict[str, str]] = []
    for msg in state.get("messages", []):
        if hasattr(msg, "type") and hasattr(msg, "content"):
            if msg.type == "human":
                history.append({"role": "user", "content": msg.content})
            elif msg.type == "ai":
                history.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, dict) and "role" in msg and "content" in msg:
            history.append({"role": str(msg["role"]), "content": str(msg["content"])})
        elif isinstance(msg, str):
            history.append({"role": "assistant", "content": msg})

    messages = [
        {"role": "system", "content": """
あなたは優秀な Azure エンジニアです。上記の要件から最小のBicepを出力します。
説明は不要、```bicep ...``` のコードブロックで返してください。
"""}
    ] + history + [
        {"role": "user", "content": "要件に基づいてBicepコードを生成してください。"}
    ]

    resp = await llm.ainvoke(messages)
    text = (resp.content or "").strip()

    code_text = text
    m = re.search(r"```(?:bicep)?\s*\n([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        code_text = m.group(1).strip()

    done_msg = "Bicepコードを生成しました。右側のエディタで確認・編集できます。"
    return {
        "messages": [{"role": "assistant", "content": done_msg}],
        "n_callings": state.get("n_callings", 0),
        "current_user_message": state.get("current_user_message", ""),
        "bicep_code": code_text,  # 完了の印
    }

# ──────────────────────────────────────────────────────────────────────────────
# グラフ構築（checkpointer 付き：SQLite → 無ければ Memory に自動フォールバック）
# ──────────────────────────────────────────────────────────────────────────────
def build_graph():
    gb = StateGraph(State)
    gb.add_node("hearing", hearing)
    gb.add_node("code_generation", code_generation)

    gb.set_entry_point("hearing")
    gb.add_conditional_edges(
        "hearing",
        should_hear_again,
        {"again": "hearing", "done": "code_generation"},
    )
    gb.set_finish_point("code_generation")

    if SqliteSaver is not None:
        checkpointer = SqliteSaver("checkpoints.db")
        if DEBUG_LOG:
            print("[graph] Using SqliteSaver(checkpoints.db)")
    elif MemorySaver is not None:
        checkpointer = MemorySaver()
        if DEBUG_LOG:
            print("[graph] Using MemorySaver (no persistence)")
    else:
        raise RuntimeError("No available checkpointer: SqliteSaver and MemorySaver are both unavailable")

    return gb.compile(checkpointer=checkpointer)

GRAPH = build_graph()

# ──────────────────────────────────────────────────────────────────────────────
# ルーティング
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Bicep Generator API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "bicep-generator-api"}

@app.get("/config")
async def get_config():
    return {
        "azure_openai_endpoint": AZURE_OPENAI_ENDPOINT,
        "azure_openai_deployment": AZURE_OPENAI_DEPLOYMENT,
        "openai_api_version": OPENAI_API_VERSION,
    }

@app.post("/reset")
async def reset_conversation(session_id: Optional[str] = "default"):
    """
    会話リセット（簡易版）
    - 実運用はフロントで毎会話UUIDを割り当て、thread_idを変更するのが最も安全
    - ここでは messages, bicep_code などを空にする
    """
    config = {"configurable": {"thread_id": session_id or "default"}}
    GRAPH.update_state(
        config,
        {"messages": [], "bicep_code": "", "n_callings": 0, "current_user_message": ""},
    )
    return {"message": f"会話({session_id})がリセットされました"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    使い方:
      1) POST /chat {"session_id":"abc","message":"ストレージがほしい"}
         → 質問が返る（is_complete=false）
      2) POST /chat {"session_id":"abc","message":"Webアプリ用です"}
         → 次の質問（is_complete=false）
      3) …繰り返し
      4) 十分な要件になると code_generation が走り、bicep_code + is_complete=true を返す
    """
    try:
        session_id = request.session_id or "default"
        config = {"configurable": {"thread_id": session_id}}

        if DEBUG_LOG:
            print(f"[chat] session={session_id} message={request.message!r}")

        # ① Humanの発話を state に追加（ある場合のみ）
        if request.message:
            GRAPH.update_state(
                config,
                {"messages": [{"role": "user", "content": request.message}]},
            )

        # ② グラフを1ステップだけ前進
        async for chunk in GRAPH.astream(None, config=config, stream_mode="updates"):
            # 最初の更新で停止（1ステップのみ実行）
            break

        # ③ 現在stateを取得
        state = GRAPH.get_state(config).values
        msgs = state.get("messages", [])
        bicep = state.get("bicep_code")

        # 直近のAI発話
        latest_ai_text = None
        for m in reversed(msgs):
            if isinstance(m, dict):
                if m.get("role") == "assistant":
                    latest_ai_text = m.get("content")
                    break
            elif hasattr(m, "type") and getattr(m, "type") == "ai":
                latest_ai_text = getattr(m, "content", None)
                break
            elif isinstance(m, str):
                latest_ai_text = m
                break

        if bicep:
            # 完了
            return ChatResponse(
                message=latest_ai_text or "Bicepコードの生成が完了しました！",
                bicep_code=bicep,
                is_complete=True,
            )

        # 継続（質問返し）
        return ChatResponse(
            message=latest_ai_text or "次の質問を用意しています…",
            bicep_code="",
            is_complete=False,
        )

    except Exception as e:
        if DEBUG_LOG:
            print("[chat] ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=f"エラーが発生しました: {str(e)}")

# ──────────────────────────────────────────────────────────────────────────────
# dev run
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # Windows環境では明示的にlocalhostを指定すると楽
    uvicorn.run(app, host="127.0.0.1", port=8000)
