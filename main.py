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
            {"role": "system", "content": "あなたは優秀な要件ヒアリング担当者です。ユーザーの要件を深掘りして、必要な情報を引き出すための質問を生成するのがあなたの役割です。"},
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
            {"role": "system", "content": "ユーザーの要件が十分にヒアリングできたら 'done' と答えてください。"},
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
