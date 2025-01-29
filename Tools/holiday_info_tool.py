from langchain_core.tools import tool


@tool("holiday_info_tool")
def holiday_info_tool(question: str) -> str:
    """
    Цей інструмент повертає інформацію про компанію "Аврора".
    """

    info = """ """

    return info
