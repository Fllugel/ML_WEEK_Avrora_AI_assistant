from langchain_core.tools import tool


@tool("avrora_info_tool")
def avrora_info_tool(topic: str) -> str:
    """
    Цей інструмент повертає інформацію про компанію "Аврора". Залежно від теми (topic), повертається різна інформація. Ось можливі теми:
    numbers - цифрова інформація про компанію така як кількість магазинів, сума сплачених податків, благодійність та інше.
    """

    if topic == "numbers":
        info = """
    **Аврора в цифрах**  
1600+ Магазинів Аврора по Україні  
6,4 млрд Гривень сплачених податків і зборів (за 2024 р.)  
498 млн Гривень на благодійну допомогу (з 24.02.2022 р.)  
2800+ Звернень про благодійну допомогу опрацьовано  
765 000+ Видано одиниць товару гуманітарної допомоги  
70% Частка вітчизняних постачальників """
    else:
        info = "Неправильний запит (topic). Спробуйте ще раз."

    return info
