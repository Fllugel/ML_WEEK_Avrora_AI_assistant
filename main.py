from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langsmith import utils
from config import MAX_MESSAGES_IN_SHORT_TERM_MEMORY
from graph import graph_builder
import sys
import uvicorn

load_dotenv(dotenv_path=".env")

utils.tracing_is_enabled()

# Compile the graph
runnable = graph_builder.compile()

app = FastAPI()

# Chat history storage
chat_history = []


class ChatRequest(BaseModel):
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
async def chat(request: ChatRequest):
    global chat_history

    # Build the input payload
    input_payload = {
        "history": chat_history,
        "input": request.input
    }

    # Invoke the graph
    response = runnable.invoke({
        "messages": prompt.format_messages(**input_payload)
    })

    # Extract the assistant's response
    response_message = response["messages"][-1].content

    # Update chat history
    chat_history.append({"role": "user", "content": request.input})
    chat_history.append({"role": "assistant", "content": response_message})

    # Enforce memory limit
    if len(chat_history) > MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:
        chat_history = chat_history[-MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:]

    return {"response": response_message}


@app.post("/clear_history")
async def clear_history():
    global chat_history
    chat_history = []
    return {"message": "Chat history cleared."}


def chat_loop():
    print("Welcome to the chatbot! Type 'exit' to quit the conversation.")

    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        input_payload = {"history": chat_history, "input": user_input}
        response = runnable.invoke({"messages": prompt.format_messages(**input_payload)})
        response_message = response["messages"][-1].content

        print(f"Bot: {response_message}")

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response_message})

        if len(chat_history) > MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:
            chat_history = chat_history[-MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        chat_loop()
