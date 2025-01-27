from graph import graph_builder

graph = graph_builder.compile()

user_input = "What is Rubix cube?"

events = graph.stream(
    {"messages": [("user", user_input)]}, stream_mode="values"
)

for event in events:
    event["messages"][-1].pretty_print()
