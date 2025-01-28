from langchain_core.prompts import ChatPromptTemplate
from config import MAX_MESSAGES_IN_SHORT_TERM_MEMORY
from graph import graph_builder


# Compile the graph
runnable = graph_builder.compile()

# Define the system prompt
system_prompt = 'Always start chat with "Hi, Im Bob"'


def chat_loop():
    print("Welcome to the chatbot! Type 'exit' to quit the conversation.")

    # Initialize the state using ChatPromptTemplate
    state = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{user_input}"),
        ("human", "{human_scratchpad}")
    ])

    # Initialize messages in the state
    messages = []

    while True:
        # Take user input
        user_input = input("You: ")

        # Exit condition
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Create a message dictionary for user input
        messages.append({"role": "user", "content": user_input})

        # Enforce the maximum number of messages in memory
        if len(messages) > MAX_MESSAGES_IN_SHORT_TERM_MEMORY:
            messages.pop(0)

        # Create the state inputs dynamically
        input_state = {
            "user_input": user_input,
            "human_scratchpad": "",
            "messages": messages
        }

        # Run the graph to get a response
        respond = runnable.invoke(input_state)

        # Get the response message from the assistant
        response_message = respond["messages"][-1].content

        print(f"Bot: {response_message}")

        # Add the assistant response to messages
        messages.append({"role": "assistant", "content": response_message})

        # Enforce the maximum number of messages in memory
        if len(messages) > MAX_MESSAGES_IN_SHORT_TERM_MEMORY:
            messages.pop(0)


# Start the chat loop
chat_loop()
