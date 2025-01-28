# Define the chat loop function
from graph import graph_builder

runnable = graph_builder.compile()


def chat_loop():
    print("Welcome to the chatbot! Type 'exit' to quit the conversation.")

    state = {
        "messages": []
    }

    while True:
        # Take user input
        user_input = input("You: ")

        # Exit condition
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Add user input to the state messages
        state["messages"].append({"role": "user", "content": user_input})

        # Run the graph to get a response
        respond = runnable.invoke(state)

        response_message = respond["messages"][-1].content

        print(f"Bot: {response_message}")

        state["messages"].append({"role": "assistant", "content": response_message})

        if user_input.lower() == "exit":
            print("Goodbye!")
            return


chat_loop()
