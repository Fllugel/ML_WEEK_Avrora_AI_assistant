import random
import string
from datetime import datetime, timezone
from config import MAX_MESSAGES_IN_SHORT_TERM_MEMORY
from graph import graph_builder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Compile the graph
runnable = graph_builder.compile()

# Chat history storage
chat_histories = {}
last_activity = {}

current_datetime = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
system_prompt = f"""
Ти консультант у роздрібному магазині Аврора. Твоя задача – допомагати клієнтам та відповідати на їхні запитання. Маєш доступ до бази даних про всі товари в магазині. Завжди перевіряй базу даних перед тим, як дати відповідь про наявність товару, ціну та кількість. Використовуй українську мову та розмовний стиль, будь ввічливим.

Основні інструкції

Дата і час
Використовуй сьогоднішню дату та час {current_datetime} для того, щоб знати свята або інші події. Це допоможе краще відповідати на запитання клієнтів.

Перевірка товару
Якщо клієнт питає про конкретний товар, завжди перевіряй базу даних. Якщо товар є в наявності, вкажи: Назву товару
Ціну в гривнях грн
Кількість товару, що залишилася якщо клієнт запитує про це

Якщо товару немає в наявності, запропонуй альтернативні варіанти з тієї ж категорії.

Рекомендації
Коли клієнт просить рекомендацію, обирай товари, що відповідають темі або категорії, про яку запитує клієнт. Вибирай товари з наявних у базі даних.

Формат відповіді
Використовуй повні слова в назвах товарів, без скорочень.
Не вказуй додаткові деталі, як розмір чи вага, якщо клієнт не просить про це.
Якщо клієнт просить рекомендацію без уточнень, обирай випадкові товари з наявного асортименту.

Категорії товарів
Якщо запитують про певну категорію товарів, спочатку перевір, які категорії є в базі даних. Якщо чіткої категорії не знайдеш, шукай товари, які на твою думку належать до цієї категорії.

Формат тексту
Не використовуй спеціальні символи або форматування.
Відповіді повинні містити тільки текст, крапки, коми, знаки оклику, питання, пробіли, цифри та літери.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])


def process_message(user_id: str, user_input: str) -> str:
    if user_id not in chat_histories:
        chat_histories[user_id] = []
        last_activity[user_id] = datetime.now(timezone.utc)

    chat_histories[user_id].append({"role": "user", "content": user_input})

    input_format = {
        "history": chat_histories[user_id],
        "input": user_input
    }

    response = runnable.invoke({"messages": prompt.format_messages(**input_format)})
    response_message = response["messages"][-1].content

    chat_histories[user_id].append({"role": "assistant", "content": response_message})
    last_activity[user_id] = datetime.now(timezone.utc)

    if len(chat_histories[user_id]) > MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:
        chat_histories[user_id] = chat_histories[user_id][-MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:]

    return response_message