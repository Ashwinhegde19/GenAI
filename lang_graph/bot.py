from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Literal

class State(TypedDict):
    user_msg : str
    ai_msg : str
    is_coding_question : bool

def detect_query(state: State):
    user_msg = state.get("user_msg")

#  make a gemini api call

    state["is_coding_question"] = True
    return state

def route_edge(state: State) -> Literal["solve_coding_question", "solve_simple_question"]:
    if state.get("is_coding_question"):
        return "solve_coding_question"
    else:
        return "solve_simple_question"

def solve_coding_question(state: State):
    user_msg = state.get("user_msg")
    state["ai_msg"] = "I can help you with that coding question. Here's a solution: ..."
    return state

def solve_simple_question(state: State):
    user_msg = state.get("user_msg")
    state["ai_msg"] = "I can help you with that simple question. Here's an answer: ..."
    return state



graph_builder = StateGraph(State)

graph_builder.add_node("detect_query", detect_query)
graph_builder.add_node("solve_coding_question", solve_coding_question)
graph_builder.add_node("solve_simple_question", solve_simple_question)


graph_builder.add_edge(START, "detect_query")
graph_builder.add_conditional_edges("detect_query", route_edge)

graph_builder.add_edge("solve_coding_question", END)
graph_builder.add_edge("solve_simple_question", END)

graph = graph_builder.compile()

def run_graph():
    state = {
        "user_msg": "Hey, How are you?",
        "ai_msg": "",
        "is_coding_question": False
    }
    result = graph.invoke(state)
    print("Final Result",result)

run_graph()