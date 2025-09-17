import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
from langchain.chat_models import init_chat_model
from langchain_openai import AzureChatOpenAI
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, AnyMessage
import dotenv

# 環境変数の読み込み
dotenv.load_dotenv()

# FastAPIアプリケーションの初期化
app = FastAPI(title="Bicep Generator API", version="1.0.0")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.jsの開発サーバー
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure OpenAI の設定
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)

# Debug flag (set DEBUG_LOG=1 to enable console prints)
DEBUG_LOG = os.getenv("DEBUG_LOG", "0") in ("1", "true", "True")

# データモデル
class ChatMessage(BaseModel):
    content: str
    sender: str
    timestamp: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    message: str
    bicep_code: str = ""
    is_complete: bool = False

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    n_callings: int
    current_user_message: str

# Helper: build OpenAI-style history safely from heterogeneous message items
def build_history(messages: List[Any]) -> List[Dict[str, str]]:
    safe: List[Dict[str, str]] = []
    for m in messages or []:
        # LangChain message objects (HumanMessage/AIMessage) have .type and .content
        if hasattr(m, "type") and hasattr(m, "content"):
            t = getattr(m, "type", "")
            role = "user" if t == "human" else ("assistant" if t == "ai" else "system")
            safe.append({"role": role, "content": getattr(m, "content", "")})
        elif isinstance(m, dict) and "role" in m and "content" in m:
            safe.append({"role": str(m.get("role")), "content": str(m.get("content", ""))})
        elif isinstance(m, str):
            # default to assistant text when role unknown
            safe.append({"role": "assistant", "content": m})
        else:
            # Fallback stringification
            safe.append({"role": "assistant", "content": str(m)})
    return safe

# LangGraphのノード関数（修正版）
async def hearing(state: State):
    """要件ヒアリング関数"""
    history = build_history(state["messages"])
    
    response = await llm.ainvoke(
        [
            {"role": "system", "content": """
             ## 役割
             あなたは優秀な要件ヒアリング担当者です。ユーザーの要件を深掘りして、必要な情報を引き出すための質問を生成するのがあなたの役割です。

             ## 目的
             ユーザーは、最終的に Azure 上に環境を構築しようとしています。ヒアリング項目はあくまでも、Azure 上に環境を構築するために必要な情報に限定してください。

             ## 考え方
             まだ、背景が明確になっていない場合には、まずは背景を明確にするための質問をしてください。

             ## 例
             以下は、一連の応答の例です。
             - hearing どのような目的や業務を実現するためにAzure環境を構築したいと考えていますか？
             - User: AppGWでのバックエンドへのルーティングについて、複雑なケースを実現可能か確認したい
             - ユーザーのビジネス要件は何ですか？
             - hearing 使用するバックエンドは何種類あり、それぞれの役割は何ですか？
             - User: 2種類、Webアプリ  
             - should_hear_again:  again
             - hearing Webアプリ間でどのようにパスを分けたいと考えていますか？
             - User: パスベースで
             - hearing 具体的なパスの構成やルールを教えていただけますか？
             - User: www.app1.com を A に、www.app2.com をBに
             """},
            *history,
            {"role": "user", "content": "ユーザーの要件を深掘りするための質問をしてください。ただし、質問は一つだけにしてください。分量は短く、簡潔にしてください。"},
        ]
    )
    
    hearing_question = response.content
    # Return assistant question as plain text; add_messages will coerce appropriately
    return {
        "messages": [hearing_question], 
        "n_callings": state["n_callings"] + 1,
        "current_user_message": state["current_user_message"]
    }

async def should_hear_again(state: State) -> str:
    """ヒアリング継続判定関数"""
    history = build_history(state["messages"])
    
    response = await llm.ainvoke(
        [
            {"role": "system", "content": """
             ## 役割
             あなたは、ヒアリング結果をもとに、Azure 上にシステムを構築するための要件が十分にヒアリングできたかどうかを判断する役割です。
             誰が考えても9割くらいは同じ環境が作れるかどうかを軸に、ユーザーの要件が十分にヒアリングできていたら 'done' と答えてください。
             """},
            *history,
            {"role": "user", "content": "要件は十分にヒアリングできましたか？ 'done' か 'again' で答えてください。"},
        ]
    )

    if state["n_callings"] >= 10:
        return "done"
        
    if "done" in response.content.lower():
        return "done"
    else:
        return "again"

async def code_generation(state: State):
    """コード生成関数"""
    history = build_history(state["messages"])
    
    response = await llm.ainvoke(
        [
            {"role": "system", "content": "あなたは優秀な Azure エンジニアです。ユーザーの要件に基づいて、Bicep でコードを生成してください。"},
            *history,
            {"role": "user", "content": "上記の要件に基づいて、Bicepでコードを生成してください。"},
        ]
    )
    
    # 生成結果からBicepコードブロックを優先的に抽出
    text = response.content or ""
    code_text = text
    try:
        # ```bicep ... ``` または ``` ... ``` を抽出
        import re
        m = re.search(r"```(?:bicep)?\n([\s\S]*?)```", text, re.IGNORECASE)
        if m:
            code_text = m.group(1).strip()
    except Exception:
        # フォールバックでそのまま
        code_text = text
    
    # チャット欄には短い完了メッセージのみを載せ、実際のBicepは bicep_code に格納
    done_msg = "Bicepコードを生成しました。右側のエディタで確認・編集できます。"
    return {
        "messages": [{"role": "assistant", "content": done_msg}],
        "n_callings": state["n_callings"],
        "bicep_code": code_text,
    }

# LangGraphの設定
def create_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("hearing", hearing)
    graph_builder.add_node("code_generation", code_generation)
    graph_builder.set_entry_point("hearing")
    graph_builder.add_conditional_edges(
        "hearing",
        should_hear_again,
        {
            "again": "hearing",
            "done": "code_generation",
        }
    )
    graph_builder.set_finish_point("code_generation")
    return graph_builder.compile()

# グローバル状態管理（実際のプロダクションではRedisやDBを使用）
conversation_sessions: Dict[str, Dict[str, Any]] = {}

@app.get("/")
async def root():
    return {"message": "Bicep Generator API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """チャットエンドポイント"""
    try:
        if DEBUG_LOG:
            print("[chat] request.message:", request.message)
        # セッション管理（簡易版）
        session_id = "default"  # 実際はユーザーIDやセッションIDを使用
        
        if session_id not in conversation_sessions:
            # 新しいセッションの初期化
            conversation_sessions[session_id] = {
                "graph": create_graph(),
                "state": State(messages=[], n_callings=0, current_user_message=""),
                "is_complete": False,
                "last_code": "",
            }
        
        session = conversation_sessions[session_id]
        
        # 既に完了している場合は新しいセッションを開始
        if session["is_complete"]:
            conversation_sessions[session_id] = {
                "graph": create_graph(),
                "state": State(messages=[], n_callings=0, current_user_message=""),
                "is_complete": False,
                "last_code": "",
            }
            session = conversation_sessions[session_id]
        
        # 現在のユーザーメッセージを状態に設定
        session["state"]["current_user_message"] = request.message
        # 会話履歴にもユーザーメッセージを積む（ロール付き）
        session["state"]["messages"].append({"role": "user", "content": request.message})

        # LangGraphの実行
        if DEBUG_LOG:
            print("[chat] invoking graph... n_callings:", session["state"]["n_callings"]) 
        result = await session["graph"].ainvoke(session["state"])
        
        # 状態の更新
        session["state"] = result
        
        # 結果の処理
        if "bicep_code" in result:
            # コード生成完了
            session["is_complete"] = True
            session["last_code"] = result["bicep_code"]
            resp = ChatResponse(
                message="Bicepコードの生成が完了しました！",
                bicep_code=session["last_code"],
                is_complete=True
            )
            if DEBUG_LOG:
                print("[chat] response: is_complete=True, code_len=", len(session["last_code"]))
            return resp
        else:
            # ヒアリング継続
            latest = result["messages"][ -1 ] if result["messages"] else None
            if latest is None:
                latest_message_text = "質問を準備中です..."
            else:
                # Extract plain text from message objects/dicts/strings
                if isinstance(latest, str):
                    latest_message_text = latest
                elif isinstance(latest, dict):
                    latest_message_text = latest.get("content", str(latest))
                else:
                    # LangChain message objects have .content
                    latest_message_text = getattr(latest, "content", str(latest))

            resp = ChatResponse(
                message=latest_message_text,
                bicep_code=session.get("last_code", ""),
                is_complete=False
            )
            if DEBUG_LOG:
                print("[chat] response: is_complete=False, msg_len=", len(latest_message_text), 
                      " code_len=", len(session.get("last_code", "")))
            return resp
            
    except Exception as e:
        if DEBUG_LOG:
            print("[chat] ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=f"エラーが発生しました: {str(e)}")

@app.post("/reset")
async def reset_conversation():
    """会話をリセット"""
    session_id = "default"
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
    return {"message": "会話がリセットされました"}

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {"status": "healthy", "service": "bicep-generator-api"}

@app.get("/config")
async def get_config():
    """現在のAzure OpenAI設定（キーは含めない）"""
    return {
        "azure_openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "azure_openai_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        "openai_api_version": os.getenv("OPENAI_API_VERSION"),
    }

if __name__ == "__main__":
    import uvicorn
    # Bind to localhost by default on Windows to avoid firewall/socket permission prompts
    uvicorn.run(app, host="127.0.0.1", port=8000)