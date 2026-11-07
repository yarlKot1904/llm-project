"""Инструменты агента. Требование FR-6."""

import re
from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
_PART = r"P-[0-9]{4}[A-Z]?"
_MODEL = r"(?:AR|BZ|GS)-[0-9]{3}"


def _schema(name: str, description: str, **properties: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


_QUESTION = {"type": "string", "description": "Вопрос по документам с исходными кодами и названиями"}
_PART_ID = {"type": "string", "pattern": f"^{_PART}$", "description": "Подтверждённый номер детали, например P-1042B"}
_WAREHOUSE = {"type": "string", "enum": WAREHOUSES}
TOOL_SCHEMAS: list[dict] = [
    _schema(
        "search_knowledge", "Поиск узла, детали, замены, поставщика или регламента по документам.", question=_QUESTION,
    ),
    _schema("summarize_topic", "Обзор по линейке, статистике отказов или нескольким моделям.", question=_QUESTION),
    _schema("check_stock", "Получить остаток детали на конкретном складе.", part_id=_PART_ID, warehouse=_WAREHOUSE),
    _schema("get_price", "Получить цену детали в рублях.", part_id=_PART_ID),
    _schema(
        "create_service_order", "Подготовить явно запрошенный заказ на указанное пользователем количество.",
        part_id=_PART_ID, warehouse=_WAREHOUSE, qty={"type": "integer", "minimum": 1},
        model={"type": "string", "pattern": f"^{_MODEL}$"},
    ),
]
_REQUIRED = {
    schema["function"]["name"]: set(schema["function"]["parameters"]["required"])
    for schema in TOOL_SCHEMAS
}


def argument_error(name: str, args: dict) -> str | None:
    """Общая проверка для прямого вызова инструмента и подготовки заказа агентом."""
    if name not in _REQUIRED:
        return "Неизвестный инструмент."
    if not isinstance(args, dict) or "_raw" in args:
        return "Аргументы инструмента не читаются."
    if args.keys() != _REQUIRED[name]:
        return f"Ожидаются аргументы: {', '.join(sorted(_REQUIRED[name]))}."
    if "question" in args and (not isinstance(args["question"], str) or not args["question"].strip()):
        return "Укажите непустой вопрос."
    if "part_id" in args and (not isinstance(args["part_id"], str) or not re.fullmatch(_PART, args["part_id"])):
        return "Укажите номер детали вида P-1042 или P-1042B."
    if "warehouse" in args and args["warehouse"] not in WAREHOUSES:
        return f"Неизвестный склад. Допустимые склады: {', '.join(WAREHOUSES)}."
    if "qty" in args and (type(args["qty"]) is not int or args["qty"] <= 0):
        return "Количество должно быть положительным целым числом."
    if "model" in args and (not isinstance(args["model"], str) or not re.fullmatch(_MODEL, args["model"])):
        return "Укажите код модели вида AR-210, BZ-120 или GS-520."
    return None


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    def search_knowledge(question: str) -> dict:
        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def summarize_topic(question: str) -> dict:
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    def checked(name: str, tool: Callable[..., dict]) -> Callable[..., dict]:
        def run(**args) -> dict:
            error = argument_error(name, args)
            if error:
                return {"error": error}
            try:
                return tool(**args)
            except ErpNotFound as exc:
                return {"error": f"В ERP нет данных: {exc}", "kind": "not_found"}
            except ErpRejected as exc:
                return {"error": f"ERP отклонила запрос: {exc}", "kind": "rejected"}

        return run

    functions = {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": erp.get_stock,
        "get_price": erp.get_price,
        "create_service_order": erp.create_order,
    }
    return {name: checked(name, tool) for name, tool in functions.items()}
