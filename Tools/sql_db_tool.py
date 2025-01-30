import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain import hub
from typing_extensions import Annotated, TypedDict
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

# Load environment variables
load_dotenv()

# Initialize the database
db = SQLDatabase.from_uri("sqlite:///Data/database.db")

# Pull the query prompt template
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")


# Define the State object structure
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    history: list


# Define the QueryOutput structure
class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]


llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.7, openai_api_key=os.getenv("GPT_API_KEY"))


# Function to write the query
def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
            "history": state["history"]
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}


# Function to execute the query
def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}


# Function to generate the answer
def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}\n'
        f'History: {state["history"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}


# Define the tool using the @tool decorator
@tool("find_data_in_db")
def find_data_in_db(question: str, history: list) -> str:
    """Create a tool that searches for data in a store database. The database contains information about products, including:
        -Category: Name and unique identification number of the category.
        -Subcategory: Name and unique identification number of the subcategory.
        -Product Name: Specific name of the product.
        -Stock Quantity: The number of units available in the store.
        -Stock Value: The total price of all units of the product in stock."""
    state = {"question": question, "history": history}
    state.update(write_query(state))
    state.update(execute_query(state))
    state.update(generate_answer(state))
    return state["answer"]