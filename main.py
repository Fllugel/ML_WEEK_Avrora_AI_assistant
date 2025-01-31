import sys
import uvicorn
import random
import string
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

# Updated CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing - change to specific origins in production
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


current_datetime = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
system_prompt = f"""Ти консультант у роздрібному магазині "Аврора". Твоя мета - допомагати клієнтам та відповідати на їх запитання.
Сьогоднішня дата та час: {current_datetime}. Можеш дату і час використовувати щоб знати яке сьогодні свято.
У тебе є доступ до бази даних про усі товари у магазині. Якщо тебе питають щось про товар в магазині ЗАВЖДИ перевіряй базу даних. ВСІ ЦІНИ В ГРН. Рекомендовані товари повинні відповідати категорії або темі, що були зазначені клієнтом.
Надавай відповіді у легкму для розуміння розмовному стилі.
БУДЬ ВІЧЛИВИМ.

Коли пишеш назву товару, використовуй ПОВНІ слова, а НЕ скорочення.

У тебе є короткострокова пам'ять. Ти можеш пам'ятати до {MAX_MESSAGES_IN_SHORT_TERM_MEMORY} повідомлень.
Не пиши кількість товару якщо не питають.

- Якщо запитали, про якийсь товар, ЗАВЖДИ дивитись базу даних, і відповідати і якщо є в наявності, то повідомити їх ціну і кількість товарів, що залишились. Ти можеш брати цю інформацію тільки з бази даних. Якщо зараз такого продукту немає в наявності, запропонуй декілька варіантів.
- Якщо запитали, рекомендувати товари, базуючись на описаній події. Рекомендуй тільки ті товари, що є в базі даних і тільки якщо їх кількість більша за 0.
- Якщо питають назву товару не пиши такі деталі як розмір, вага.
- Якщо питають рекомендацію/що купити то рекомендуй конкретні випадкові ТОВАРИ з магазину.
- Якщо питають про певну катигорію спочатку взнай які в тебе є катигорії товарів, а потім відповідай.Якщо чітко таку катигорію в базі даних не виходить знайти шукай товари які на твою думку можуть належати цій категорії.

  НЕ ВИКОРИСТОВУЙ MARKDOWN. НЕ ПИШИ СПЕЦІАЛЬНІ СИМВОЛИ. ВИКОРИСТОВУЙ ЛИШЕ ТЕКСТ, КРАПКИ,КОМИ,ЗНАК ОКЛИКУ,ЗНАК ПИТАННЯ, ПРОБІЛИ, ЦИФРИ ТА ЛІТЕРИ.
  ВІДПОВІДАЙ В ПРОСТОМУ РОЗМОВНОМУ СТИЛІ.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    user_id = request.user_id
    if user_id not in chat_histories:
        chat_histories[user_id] = []
    last_activity[user_id] = datetime.now(timezone.utc)

    # Build the input payload
    input_payload = {
        "history": chat_histories[user_id],
        "input": request.input
    }

    # Invoke the graph
    response = runnable.invoke({
        "messages": prompt.format_messages(**input_payload)
    })

    # Extract the assistant's response
    response_message = response["messages"][-1].content

    # Update chat history
    chat_histories[user_id].append({"role": "user", "content": request.input})
    chat_histories[user_id].append({"role": "assistant", "content": response_message})

    # Enforce memory limit
    if len(chat_histories[user_id]) > MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:
        chat_histories[user_id] = chat_histories[user_id][-MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:]

    return {"response": response_message}


@app.post("/clear_history")
async def clear_history(request: ChatRequest, background_tasks: BackgroundTasks):
    user_id = request.user_id
    if user_id in chat_histories:
        chat_histories[user_id] = []
    print(f"Clearing history for user_id: {user_id}")
    return {"message": "Chat history cleared."}


def chat_loop():
    print("Welcome to the chatbot! Type 'exit' to quit the conversation.")

    user_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    chat_histories[user_id] = []
    last_activity[user_id] = datetime.now(timezone.utc)

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            del chat_histories[user_id]
            del last_activity[user_id]
            break

        input_payload = {"history": chat_histories[user_id], "input": user_input}
        response = runnable.invoke({"messages": prompt.format_messages(**input_payload)})
        response_message = response["messages"][-1].content

        print(f"Bot: {response_message}")

        chat_histories[user_id].append({"role": "user", "content": user_input})
        chat_histories[user_id].append({"role": "assistant", "content": response_message})
        last_activity[user_id] = datetime.now(timezone.utc)

        if len(chat_histories[user_id]) > MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:
            chat_histories[user_id] = chat_histories[user_id][-MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:]


if __name__ == "__main__":
    # Create SSL context
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    try:
        ssl_context.load_cert_chain('cert.pem', keyfile='key.pem')
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000, 
            ssl=ssl_context,
            log_level="debug"
        )
    except Exception as e:
        print(f"Failed to load SSL certificates: {e}")
        # Fallback to HTTP
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            log_level="debug"
        )
