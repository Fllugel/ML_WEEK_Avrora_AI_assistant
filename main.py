from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langsmith import utils
from config import MAX_MESSAGES_IN_SHORT_TERM_MEMORY
from graph import graph_builder

load_dotenv(dotenv_path=".env")

utils.tracing_is_enabled()

# Compile the graph
runnable = graph_builder.compile()

system_prompt = f"""You are an assistant in a retail shop called "Avrora". Your goal is to help the customer and answer their questions.
You have a database of all the products that are left in the shop. You have the category and name of each product, the number of items left, and their total cost.
A customer will ask some questions that you need to answer.
The main criteria: the  recommended products must correspond to the category or topic that was mentioned. If there are still different options choose the cheapest products.
Your responsibility: Provide a clear list of the chosen products together with their cost, and the number of items. Give the answer in the form of a recommendation that is easy
to understand. 

You have short term memory. You can remember up to {MAX_MESSAGES_IN_SHORT_TERM_MEMORY} messages. If you have more than {MAX_MESSAGES_IN_SHORT_TERM_MEMORY} messages, you will forget the oldest ones.

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


def chat_loop():
    print("Welcome to the chatbot! Type 'exit' to quit the conversation.")

    # Initialize chat history
    chat_history = []

    while True:
        # Take user input
        user_input = input("You: ")

        # Exit condition
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Build the input payload for the graph
        input_payload = {
            "history": chat_history,
            "input": user_input
        }

        # Invoke the graph
        response = runnable.invoke({
            "messages": prompt.format_messages(**input_payload)
        })

        # Extract the assistant's response
        response_message = response["messages"][-1].content

        # Display the bot's response
        print(f"Bot: {response_message}")

        # Update chat history
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response_message})

        # Enforce the maximum number of messages
        if len(chat_history) > MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:
            chat_history = chat_history[-MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:]


# Start the chat loop
chat_loop()
