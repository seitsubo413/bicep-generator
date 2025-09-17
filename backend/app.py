import os
import re
import tempfile
import subprocess
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
DEBUG_LOG = True
MAX_HEARING_CALLS = int(os.getenv("MAX_HEARING_CALLS", "20"))  # ヒアリングの最大回数
MAX_REGEN_CALLS = int(os.getenv("MAX_REGEN_CALLS", "5"))  # コード再生成（lint後）の最大回数

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
    bicep_code: str  # 最新生成コード
    lint_output: str  # lint 結果
    validation_passed: bool  # lint 合格
    code_regen_count: int  # 再生成回数（初回生成は含めない）
    phase: str  # hearing | code_generation | code_validation | completed


class ChatMessage(BaseModel):
    content: str
    sender: str
    timestamp: Optional[str] = None  # 使わない場合は省略可


class ChatRequest(BaseModel):
    session_id: Optional[str] = "default"  # フロントから会話IDを渡すのが推奨
    message: Optional[str] = None  # 空の場合は「AIの次のステップだけ」進める
    conversation_history: List[ChatMessage] = []  # 未使用（保持はcheckpointerに任せる）


class ChatResponse(BaseModel):
    message: str
    bicep_code: str = ""
    is_complete: bool = False
    phase: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# ノード定義（プロンプトはそのまま）
# ──────────────────────────────────────────────────────────────────────────────
def _to_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: List[str] = []
        for r in raw:
            if isinstance(r, str):
                parts.append(r)
            elif isinstance(r, dict):
                # Common OpenAI message part patterns
                val = r.get("text") or r.get("content") or ""
                if isinstance(val, list):
                    parts.append(" ".join(str(v) for v in val))
                else:
                    parts.append(str(val))
            else:
                parts.append(str(r))
        return "\n".join(p for p in parts if p)
    return str(raw)


def _history_from_state(state: State) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    for msg in state.get("messages", []):
        role = None
        content_val = None
        if hasattr(msg, "type") and hasattr(msg, "content"):
            if msg.type == "human":
                role = "user"
            elif msg.type == "ai":
                role = "assistant"
            content_val = getattr(msg, "content", "")
        elif isinstance(msg, dict):
            role = str(msg.get("role"))
            content_val = msg.get("content")
        elif isinstance(msg, str):
            role = "assistant"
            content_val = msg
        if role and content_val is not None:
            history.append({"role": role, "content": _to_text(content_val)})
    return history


async def hearing(state: State):
    """要件ヒアリング: 1問だけ短く投げる"""
    history = _history_from_state(state)
    messages = [
        {
            "role": "system",
            "content": """
             ## 役割
             あなたは優秀な要件ヒアリング担当者です。ユーザーの要件を深掘りして、必要な情報を引き出すための質問を生成するのがあなたの役割です。
             
             ## 目的
             ユーザーは、最終的に Azure 上に環境を構築しようとしています。ヒアリング項目はあくまでも、Azure 上に環境を構築するために必要な情報に限定してください。
             
             ## 考え方
             まだ、背景が明確になっていない場合には、まずは背景を明確にするための質問をしてください。
             """,
        },
        *history,
        {
            "role": "user",
            "content": "ユーザーの要件を深掘りするための質問をしてください。ただし、質問は一つだけにしてください。分量は短く、簡潔にしてください。",
        },
    ]
    resp = await llm.ainvoke(messages)
    question = _to_text(resp.content).strip() or "要件をもう少し教えてください。"
    return {
        "messages": [{"role": "assistant", "content": question}],
        "n_callings": state.get("n_callings", 0) + 1,
        "current_user_message": state.get("current_user_message", ""),
        "bicep_code": state.get("bicep_code", ""),
        "lint_output": state.get("lint_output", ""),
        "validation_passed": state.get("validation_passed", False),
        "phase": "hearing",
    }


def should_hear_again(state: State) -> str:
    """ヒアリング継続判定: 'done' or 'again' を返す（同期関数）"""
    history = _history_from_state(state)
    messages = [
        {
            "role": "system",
            "content": """
あなたは、Azure環境を誰が作っても9割同じになる程度に要件が満ちたか判定します。
満ちていれば 'done'、不足していれば 'again'。回答は 'done' または 'again' のみ。
""",
        },
        *history,
        {"role": "user", "content": "要件は十分ですか？ 'done' か 'again' で答えてください。"},
    ]
    if state.get("n_callings", 0) >= MAX_HEARING_CALLS:
        return "done"
    resp = llm.invoke(messages)
    ans = _to_text(resp.content).strip().lower()
    return "done" if "done" in ans else "again"


async def code_generation(state: State):
    """Bicep コード生成 (lint 結果があれば考慮)"""
    history = _history_from_state(state)
    lint_output = state.get("lint_output")
    if lint_output:
        history.append({"role": "user", "content": f"直前 lint 結果を踏まえ修正してください:\n{lint_output[:1500]}"})
    prev_code_exists = bool(state.get("bicep_code"))
    regen_count = state.get("code_regen_count", 0) + (1 if prev_code_exists else 0)
    messages = [
        {
            "role": "system",
            "content": """
あなたは優秀な Azure エンジニアです。上記の要件から最小で妥当な Bicep を出力します。
説明は不要、```bicep ...``` のコードブロックのみで返してください。直前に lint 指摘があった場合は修正してください。
""",
        },
        *history,
        {"role": "user", "content": "要件に基づいてBicepコードを生成してください。"},
    ]
    resp = await llm.ainvoke(messages)
    text = _to_text(resp.content).strip()
    code_text = text
    m = re.search(r"```(?:bicep)?\s*\n([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        code_text = m.group(1).strip()
    done_msg = "Bicepコードを生成しました。lint 検証に進みます。"
    return {
        "messages": [{"role": "assistant", "content": done_msg}],
        "n_callings": state.get("n_callings", 0),
        "current_user_message": state.get("current_user_message", ""),
        "bicep_code": code_text,
        "lint_output": "",
        "validation_passed": False,
        "code_regen_count": regen_count,
        "phase": "code_generation",
    }


async def code_validation(state: State):
    code = state.get("bicep_code", "")
    if not code:
        return {
            "messages": [{"role": "assistant", "content": "検証対象コードがありません。"}],
            "lint_output": "(no code)",
            "validation_passed": False,
        }
    tmp_path = None
    output = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bicep", mode="w", encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name
        try:
            proc = subprocess.run(
                " ".join(["az", "bicep", "lint", "--file", tmp_path]),
                capture_output=True,
                text=True,
                timeout=60,
                shell=True,
            )
            output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        except FileNotFoundError:
            output = "Azure CLI が見つかりません。Azure CLI と Bicep 拡張のインストールを確認してください。"
        except subprocess.TimeoutExpired:
            output = "az bicep lint がタイムアウトしました。"
        except Exception as e:  # noqa
            output = f"lint 実行エラー: {e}"
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    print("[lint output]", output)  # for debug
    preview = output.strip()
    if len(preview) > 1200:
        preview = preview[:1200] + "... (truncated)"
    return {
        "messages": [{"role": "assistant", "content": f"lint 結果:\n{preview}\n再生成が必要か判定します。"}],
        "lint_output": output,
        "validation_passed": False,
        "phase": "code_validation",
    }


def should_regenerate_code(state: State) -> str:
    code = state.get("bicep_code", "")
    lint_output = state.get("lint_output", "")
    # 上限到達で強制終了
    if state.get("code_regen_count", 0) >= MAX_REGEN_CALLS:
        return "ok"
    if not code:
        return "regenerate"
    messages = [
        {
            "role": "system",
            "content": """
あなたは Bicep の品質ゲート判定者です。致命的または重要な問題があれば 'regenerate'、無ければ 'ok' のみで答えてください。""",
        },
        {
            "role": "user",
            "content": f"""## Code\n```bicep\n{code}\n```\n\n## Lint Output\n{lint_output}\n""",
        },
    ]
    try:
        resp = llm.invoke(messages)
        answer = _to_text(resp.content).strip().lower()
        return "regenerate" if "regenerate" in answer else "ok"
    except Exception:
        return "regenerate"


async def finalize_validation(state: State):
    regen_count = state.get("code_regen_count", 0)
    lint_output = state.get("lint_output", "")
    base_msg = "Bicepコードは lint 検証を通過しました。"
    if regen_count >= MAX_REGEN_CALLS:
        base_msg = (
            "自動再生成の上限 (MAX_REGEN_CALLS) に達したため処理を終了しました。"
            " なお、残存する警告/エラーがある場合は手動で修正してください。"
        )
    return {
        "messages": [{"role": "assistant", "content": base_msg}],
        "validation_passed": True,
        "lint_output": lint_output,
        "phase": "completed",
    }


# ──────────────────────────────────────────────────────────────────────────────
# グラフ構築（checkpointer 付き：SQLite → 無ければ Memory に自動フォールバック）
# ──────────────────────────────────────────────────────────────────────────────
def build_graph():
    gb = StateGraph(State)
    gb.add_node("hearing", hearing)
    gb.add_node("code_generation", code_generation)
    gb.add_node("code_validation", code_validation)
    gb.add_node("finalize_validation", finalize_validation)

    gb.set_entry_point("hearing")
    gb.add_conditional_edges(
        "hearing",
        should_hear_again,
        {"again": "hearing", "done": "code_generation"},
    )
    gb.add_edge("code_generation", "code_validation")
    gb.add_conditional_edges(
        "code_validation", should_regenerate_code, {"regenerate": "code_generation", "ok": "finalize_validation"}
    )
    gb.set_finish_point("finalize_validation")

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
        {
            "messages": [],
            "bicep_code": "",
            "n_callings": 0,
            "current_user_message": "",
            "lint_output": "",
            "validation_passed": False,
            "code_regen_count": 0,
            "phase": "hearing",
        },
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
        bicep = state.get("bicep_code") or ""
        passed = state.get("validation_passed")
        phase = state.get("phase", "")

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

        if phase == "completed" or (bicep and passed):
            # 検証完了後
            return ChatResponse(
                message=latest_ai_text or "Bicepコードの生成が完了しました！",
                bicep_code=bicep or "",
                is_complete=True,
                phase=phase or ("completed" if passed else ""),
            )

        # 継続（質問返し）
        return ChatResponse(
            message=latest_ai_text or "次の質問を用意しています…",
            bicep_code=bicep if (phase in ("code_generation", "code_validation") and bicep) else "",
            is_complete=False,
            phase=phase or ("code_generation" if bicep else "hearing"),
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
