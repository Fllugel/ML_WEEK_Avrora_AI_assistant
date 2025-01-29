import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from tools import tools

# Load environment variables
load_dotenv()

# Define the system prompt
system_prompt = """You are an assistant in a retail shop called "Avrora". Your goal is to help the customer and answer their questions.
You have a database of all the products that are left in the shop. You have the category and name of each product, the number of items left, and their total cost.
A customer will ask some questions that you need to answer.
The main criteria: the  recommended products must correspond to the category or topic that was mentioned. If there are still different options choose the cheapest products.
Your responsibility: Provide a clear list of the chosen products together with their cost, and the number of items. Give the answer in the form of a recommendation that is easy
to understand.

You can:
 - If asked answer whether a specific kind of product is in stock and if so - the number of items left and their cost. You have to take this information from the database only.
If no such product is available at the time,  suggest some options (up to 3) that may be suitable, for example, from the same category.
 - If asked recommend products based on the event described. Recommend only those items that are in the table, and the number of items left is above zero.
 - If the maximum total cost was specified, recommend the items whose cost in total doesn't exceed the amount mentioned.
 - Additionally, recommend several items that may be of interest to the customer based on the holiday that is celebrated today."""

# Create the ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

# Define State object structure
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Set up the LLM with tools binding
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.7, openai_api_key=os.getenv("GPT_API_KEY"))
llm_with_tools = llm.bind_tools(tools=tools)

# Define the chatbot response function
def chatbot(state: State):
    # Format the messages with the user input
    formatted_messages = prompt.format_messages(**state)
    # Invoke the model with the current state of the conversation
    return {"messages": [llm_with_tools.invoke(formatted_messages)]}