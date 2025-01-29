from langchain_core.tools import tool


@tool("avrora_info_tool")
def avrora_info_tool(question: str) -> str:
    """
    Цей інструмент повертає інформацію про компанію "Аврора".
    """

    info = """ """

    return info
