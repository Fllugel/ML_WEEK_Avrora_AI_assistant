# Define the chat loop function
from graph import graph_builder


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
        events = graph_builder.compile().stream(state, stream_mode="values")

        for event in events:
            response_message = event["messages"][-1].content  
            print(f"Bot: {response_message}")

            # Add bot response to state messages
            state["messages"].append({"role": "assistant", "content": response_message})

            # If user wants to continue, loop again; if not, exit
            if user_input.lower() == "exit":
                print("Goodbye!")
                return


chat_loop()
