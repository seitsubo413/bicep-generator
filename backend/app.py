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
    requirement_summary: str  # ヒアリング要件サマリ（code_generation プロンプト用）


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
    requirement_summary: str = ""


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
        "requirement_summary": state.get("requirement_summary", ""),
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
    requirement_summary = state.get("requirement_summary") or "(要件サマリ未生成)"
    lint_output = state.get("lint_output") or "(lint 結果なし)"
    prev_code_exists = bool(state.get("bicep_code"))
    regen_count = state.get("code_regen_count", 0) + (1 if prev_code_exists else 0)

    messages = [
        {
            "role": "system",
            "content": """
あなたは熟練した Azure インフラエンジニアです。以下の『要件サマリ』と（存在すれば）『直近の lint 結果』のみを根拠に、最小で妥当かつ再利用しやすい Bicep コードを 1 ファイル分だけ提示してください。
出力は説明を含めず、```bicep で始まるコードブロックのみです。以前の会話内容は参照できない前提で、要件サマリ内の情報だけで不足があれば合理的なデフォルトを仮定してください。
""",
        },
        {
            "role": "user",
            "content": f"""## 要件サマリ\n{requirement_summary}\n\n## 直近 lint 結果\n{lint_output[:1500]}\n\n上記を踏まえて Bicep コードを生成してください。説明・コメントは最小限（無くても良い）。コードブロックのみで返してください。""",
        },
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
        # lint_output は "最新の lint 結果" を保持し続ける（次回再生成プロンプトに活用）
        "lint_output": state.get("lint_output", ""),
        "validation_passed": False,
        "code_regen_count": regen_count,
        "phase": "code_generation",
        "requirement_summary": requirement_summary,
    }


async def summarize_requirements(state: State):
    """ヒアリング会話全体から要件サマリを生成し state に格納する。"""
    history = _history_from_state(state)
    # 会話履歴を文字列化（長すぎる場合は適度にトリム）
    joined = []
    for h in history:
        role = h.get("role")
        content = h.get("content", "")
        if role == "user":
            joined.append(f"[USER] {content}")
        elif role == "assistant":
            joined.append(f"[ASSISTANT] {content}")
    raw_dialogue = "\n".join(joined)
    if len(raw_dialogue) > 8000:
        raw_dialogue = raw_dialogue[-8000:]

    messages = [
        {
            "role": "system",
            "content": """
あなたは Azure 向けインフラ要件の要約アシスタントです。これまでのヒアリング会話ログから、Bicep コード生成に必要な要件だけを抽出し、以下のセクション構成で簡潔にまとめてください。

出力フォーマット:
Purpose: ...
Resources: 箇条書き (種類 / 個数 / SKU / 目的)
Networking: VNet, Subnet, DNS, Public IP, NSG, Endpoint など
Security: RBAC, KeyVault, Secrets, Encryption, NetworkRules など
Scaling & Availability: スケール要件 / 可用性ゾーン / SLA 想定
Constraints: 地域, ネーミング規則, コスト制約, タグ方針
Other: 追加の注意点（無ければ 'None'）

禁止事項: コード出力や余計な説明。""",
        },
        {
            "role": "user",
            "content": f"以下がヒアリング会話ログです。要件を抽出し指示フォーマットでまとめてください。\n\n{raw_dialogue}",
        },
    ]
    resp = await llm.ainvoke(messages)
    summary_text = _to_text(resp.content).strip()
    if not summary_text:
        summary_text = "(要約生成に失敗しました)"
    # 要約そのものをユーザーに表示したいのでメッセージ本文に含める
    display_msg = f"要件を集約しました。直ちにコード生成へ進みます。\n\n===== 要件サマリ =====\n{summary_text}"
    return {
        "messages": [{"role": "assistant", "content": display_msg}],
        "requirement_summary": summary_text,
        # 特別な中間フェーズ名を付け、フロント要求入力を待たず自動前進させるために chat_endpoint で判定
        "phase": "summarizing",
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
    gb.add_node("summarize_requirements", summarize_requirements)
    gb.add_node("code_generation", code_generation)
    gb.add_node("code_validation", code_validation)
    gb.add_node("finalize_validation", finalize_validation)

    gb.set_entry_point("hearing")
    gb.add_conditional_edges(
        "hearing",
        should_hear_again,
        {"again": "hearing", "done": "summarize_requirements"},
    )
    # 要約後にコード生成
    gb.add_edge("summarize_requirements", "code_generation")
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
    config = {"configurable": {"thread_id": session_id or "default"}}  # type: ignore[assignment]
    GRAPH.update_state(  # type: ignore[arg-type]
        config,  # type: ignore[arg-type]
        {
            "messages": [],
            "bicep_code": "",
            "n_callings": 0,
            "current_user_message": "",
            "lint_output": "",
            "validation_passed": False,
            "code_regen_count": 0,
            "phase": "hearing",
            "requirement_summary": "",
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
        config = {"configurable": {"thread_id": session_id}}  # type: ignore[assignment]

        if DEBUG_LOG:
            print(f"[chat] session={session_id} message={request.message!r}")

        # ① Humanの発話を state に追加（ある場合のみ）
        if request.message:
            GRAPH.update_state(  # type: ignore[arg-type]
                config,  # type: ignore[arg-type]
                {"messages": [{"role": "user", "content": request.message}]},
            )

        # ② 必要に応じて複数ステップ自動実行（summarize_requirements -> code_generation を一気に進める）
        auto_progress_limit = 5
        steps_executed = 0
        while True:
            steps_executed += 1
            async for _chunk in GRAPH.astream(None, config=config, stream_mode="updates"):  # type: ignore[arg-type]
                break
            state = GRAPH.get_state(config).values  # type: ignore[arg-type]
            phase_now = state.get("phase")
            # summarizing フェーズはユーザー入力不要なので続行
            if phase_now == "summarizing" and steps_executed < auto_progress_limit:
                continue
            # それ以外は 1 ステップで停止
            break

        # ③ 現在stateを取得（上の while ですでに取得済みだが明示的に変数保持）
        state = GRAPH.get_state(config).values  # type: ignore[arg-type]
        msgs = state.get("messages", [])
        bicep = state.get("bicep_code") or ""
        passed = state.get("validation_passed")
        phase = state.get("phase", "")
        requirement_summary = state.get("requirement_summary", "")

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
                requirement_summary=requirement_summary,
            )

        # 継続（質問返し）
        return ChatResponse(
            message=latest_ai_text or "次の質問を用意しています…",
            bicep_code=bicep if (phase in ("code_generation", "code_validation") and bicep) else "",
            is_complete=False,
            phase=phase or ("code_generation" if bicep else "hearing"),
            requirement_summary=requirement_summary,
        )

    except Exception as e:  # noqa
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
