import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
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

tool_prompt = PromptTemplate.from_template("""
Given an input question, create a syntactically correct {dialect} query to run to help find the answer. Unless the user specifies in his question a specific number of examples they wish to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
When searching for a specific product, and you do not find the product in the database, use multiple synonyms or rephrased versions of the product name in query. Perform search queries using variations such as the first letter capitalized, the first letter in lowercase, and the entire name in uppercase letters. If a client asks for a product recommendation without a detailed description, choose a random product from the database using RANDOM. 

ALWAYS make queries in noun infinitive form when searching for a specific product. 

Only use the following tables:
{table_info}

Question: {input}
""")

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

    prompt = tool_prompt.invoke(
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
    """Tool for searching the store's database. The database contains the following data about products: product article, product category, product sub-category, product name, available quantity in the store, and price per unit."""
    state = {"question": question, "history": history}
    state.update(write_query(state))
    state.update(execute_query(state))
    state.update(generate_answer(state))
    return state["answer"]
