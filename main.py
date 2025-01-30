import sys
import uvicorn
import asyncio
from datetime import datetime, timezone
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langsmith import utils
from config import MAX_MESSAGES_IN_SHORT_TERM_MEMORY
from graph import graph_builder

sys.setrecursionlimit(1500)  # Increase the recursion limit

load_dotenv(dotenv_path=".env")

utils.tracing_is_enabled()

# Compile the graph
runnable = graph_builder.compile()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chat history storage
chat_histories = {}
last_activity = {}


class ChatRequest(BaseModel):
    user_id: str
    input: str


system_prompt = f"""Ти асистент у роздрібному магазині "Аврора". Твоя мета - допомагати клієнтам та відповідати на їх запитання.
У тебе є база даних про усі продукти, що залишились у магазині. У тебе є категорія та назва кожного товару, кількість товарів, які залишились та їх ціна.
Покупець буде ставити запитання, на які ти маєш дати відповідь.
Головний критерій: рекомендовані товари повинні відповідати категорії або темі, що були зазначені клієнтом. Якщо все ще є декілька варіантів, обирай найдешевші продукти.
Твоє завдання: Надати чіткий перелік обраних товарів, назву СКОРОТИТИ і СПРОСТИТИ і прибрати артикул, разом з їх ціною та кількістю товарів. Надай відповідь у формі легкої для розуміння рекомендації.

У тебе є короткострокова пам'ять. Ти можеш пам'ятати до {MAX_MESSAGES_IN_SHORT_TERM_MEMORY} повідомлень. Якщо у тебе є більше ніж {MAX_MESSAGES_IN_SHORT_TERM_MEMORY} повідомлень, ти забудеш найстаріші з них.

Ти можеш:
  - Якщо запитали, чи конкретний вид товару є в наявності, ЗАВЖДИ дивитись базу даних, і відповідати і якщо є в наявності, то повідомити їх ціну і кількість товарів, що залишились. Ти можеш брати цю інформацію тільки з бази даних. Якщо зараз такого продукту немає в наявності, запропонуй декілька варіантів (до 3), які можуть підійти покупцю (наприклад,з тієї ж категорії).
  - Якщо запитали, рекомендувати товари, базуючись на описаній події. Рекомендуй тільки ті товари, що є в таблиці і тільки якщо їх кількість більша за 0.
  - Якщо була вказана максимальна сумарна вартість, то рекомендуй товари, сумарна вартість яких не перевищує вказану.
  - Додатково, ти можеш рекомендувати декілька товарів, що можуть бути цікаві покупцеві, грунтуючись на тому, яке сьогодні свято.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])


@app.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    user_id = request.user_id
    print(f"Received chat request from user_id: {user_id}")
    if user_id not in chat_histories:
        chat_histories[user_id] = []
    last_activity[user_id] = datetime.now(timezone.utc)

    # Build the input payload
    input_payload = {
        "history": chat_histories[user_id],
        "input": request.input
    }
    print(f"Input payload: {input_payload}")

    # Invoke the graph
    response = runnable.invoke({
        "messages": prompt.format_messages(**input_payload)
    })
    print(f"Graph response: {response}")

    # Extract the assistant's response
    response_message = response["messages"][-1].content
    print(f"Assistant's response: {response_message}")

    # Update chat history
    chat_histories[user_id].append({"role": "user", "content": request.input})
    chat_histories[user_id].append({"role": "assistant", "content": response_message})

    # Enforce memory limit
    if len(chat_histories[user_id]) > MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:
        chat_histories[user_id] = chat_histories[user_id][-MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:]

    # Add background task to clean up inactive users
    background_tasks.add_task(cleanup_all_users)

    return {"response": response_message}


@app.post("/clear_history/{user_id}")
async def clear_history(user_id: str, background_tasks: BackgroundTasks):
    print(f"Received clear history request from user_id: {user_id}")
    if user_id in chat_histories:
        chat_histories[user_id] = []
    return {"message": "Chat history cleared."}


async def cleanup_all_users():
    while True:
        print("Clearing all users' chat history...")
        chat_histories.clear()
        last_activity.clear()
        await asyncio.sleep(300)  # Sleep for 5 minutes (300 seconds)


def chat_loop():
    print("Welcome to the chatbot! Type 'exit' to quit the conversation.")

    user_id = input("Enter your user ID: ")
    if user_id not in chat_histories:
        chat_histories[user_id] = []
    last_activity[user_id] = datetime.now(timezone.utc)

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        input_payload = {"history": chat_histories[user_id], "input": user_input}
        print(f"User input: {user_input}")
        response = runnable.invoke({"messages": prompt.format_messages(**input_payload)})
        response_message = response["messages"][-1].content
        print(f"Bot response: {response_message}")

        chat_histories[user_id].append({"role": "user", "content": user_input})
        chat_histories[user_id].append({"role": "assistant", "content": response_message})
        last_activity[user_id] = datetime.now(timezone.utc)

        if len(chat_histories[user_id]) > MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:
            chat_histories[user_id] = chat_histories[user_id][-MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        chat_loop()
