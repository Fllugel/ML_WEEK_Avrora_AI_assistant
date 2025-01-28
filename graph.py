import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import tools

# Load environment variables
load_dotenv()


# Define State object structure
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Initialize the graph builder
graph_builder = StateGraph(State)

# Set up the LLM with tools binding
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.7, openai_api_key=os.getenv("GPT_API_KEY"))
llm_with_tools = llm.bind_tools(tools=tools)


# Define the chatbot response function
def chatbot(state: State):
    # Invoke the model with the current state of the conversation
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Add the chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Add the tools node to the graph
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Add conditional edges for tool use
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Define entry and exit edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
