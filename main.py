import os
from langchain.chat_models import init_chat_model

from langchain_openai import AzureChatOpenAI
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, AnyMessage
import sys

import dotenv
dotenv.load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    n_callings: int

# Node: 要件ヒアリング
def hearing(state: State):
    history = [msg for msg in state["messages"] if msg.type in ["human", "ai"]]
    response = llm.invoke(
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
    print('hearing', hearing_question)

    try:
        user_input = input("User: ")
    except:
        sys.exit(1)

    return {"messages": [hearing_question, user_input], "n_callings": state["n_callings"] + 1}


def should_hear_again(state: State) -> str:
    history = [msg for msg in state["messages"] if msg.type in ["human", "ai"]]
    response = llm.invoke(
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
    print('should_hear_again: ', response.content)

    if state["n_callings"] >= 10:
        return "done"
        
    if "done" in response.content.lower():
        return "done"
    else:
        return "again"
    
# Node: コード生成
def code_generation(state: State):
    history = [msg for msg in state["messages"] if msg.type in ["human", "ai"]]
    response = llm.invoke(
        [
            {"role": "system", "content": "あなたは優秀な Azure エンジニアです。ユーザーの要件に基づいて、Bicep でコードを生成してください。"},
            *history,
            {"role": "user", "content": "上記の要件に基づいて、Bicepでコードを生成してください。"},
        ]
    )
    print('code:', response.content)
    return {"messages": response.content, "n_callings": state["n_callings"]}

# ノードの追加
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

app = graph_builder.compile()

initial_state = State(n_callings=0)
result = app.invoke(initial_state)
print(result)
