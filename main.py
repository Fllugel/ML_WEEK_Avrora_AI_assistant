import os
from dotenv import load_dotenv
from config import MAX_MESSAGES_IN_SHORT_TERM_MEMORY
from graph import graph_builder
from Nodes.chatbot_node import prompt  # Import the prompt from chatbot_node

load_dotenv()

if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGSMITH_TRACING"] = "true"

# Compile the graph
runnable = graph_builder.compile()

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