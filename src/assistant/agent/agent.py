"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable
from decimal import Decimal

from assistant.agent.tools import TOOL_SCHEMAS, argument_error
from assistant.graphrag.search import REFUSAL
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import find_codes
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

_SYSTEM = (
    "Ты ассистент сервисной службы. Сначала получи факты из инструментов. "
    "Для конкретного вопроса по документам используй search_knowledge, для обзора по линейке и статистики "
    "используй summarize_topic. В поисковом вопросе сохрани исходные коды и названия. "
    "Если номер детали не дан, найди актуальную деталь в документах, прежде чем обращаться в ERP. "
    "Цену получай через get_price, остаток через check_stock. Если запрошено и то и другое, вызови оба инструмента. "
    "«Какую деталь заказать» означает рекомендацию. create_service_order вызывай при явном запросе оформления. "
    "Для заказа бери модель, склад и количество из вопроса, актуальную деталь из вопроса или поиска. "
    "Отвечай кратко, числа пиши цифрами, цену в рублях (руб.), остаток в штуках (шт.). "
    "При отсутствии данных скажи «В базе знаний не найдено». Остаток 0 означает, что деталь закончилась. "
    "Сохраняй различие между отсутствием данных и недоступностью ERP."
)
_ERP_UNAVAILABLE = "Данные ERP получить не удалось."
_ORDER_REQUEST = re.compile(
    r"\b(?:(?:оформи(?:те)?|создай(?:те)?|подготовь(?:те)?)\s+заказ|закажи(?:те)?"
    r"|(?:хочу|прошу)\s+(?:заказать|оформить заказ))\b",
    re.IGNORECASE,
)
_ORDER_INSTRUCTION = (
    " Пользователь просит оформить заказ. Получив подтверждённый номер детали, "
    "вызови create_service_order с количеством, складом и моделью из вопроса. "
    "Если номер ещё неизвестен, вызови search_knowledge. "
    "Не запрашивай остаток и цену, если пользователь этого не просил."
)
_ORDER_SCHEMA = [schema for schema in TOOL_SCHEMAS if schema["function"]["name"] == "create_service_order"]
_AMOUNT = r"[−-]?(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?"
_MEASURE = re.compile(rf"(?<![\w.-])({_AMOUNT})\s*(₽|руб\w*|шт\b|штук\w*|единиц\w*)", re.IGNORECASE)
_LABELED = re.compile(rf"\b(цена|стоимость|остаток)\s*(?:[:=—–-]\s*)?({_AMOUNT})(?![\w.,-])", re.IGNORECASE)
_ORDER = re.compile(r"\bSO-\d+\b", re.IGNORECASE)
_SMALL_NUMBERS = {
    "один": 1, "одну": 1, "одна": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5,
    "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
}
_QTY_VALUE = r"(?<![\w.,-])(?:[1-9]\d*|" + "|".join(_SMALL_NUMBERS) + r")(?![\w.,-])"
_QTY_UNIT = re.compile(rf"({_QTY_VALUE})\s+(?:шт\b|штук\w*|единиц\w*|детал\w*|экземпляр\w*)", re.IGNORECASE)
_QTY_LABEL = re.compile(
    rf"\b(?:количество|qty|закажи(?:те)?|заказ(?:ать)?|закажите|оформи(?:те)? заказ)"
    rf"\s*[:=]?\s+({_QTY_VALUE})(?!\s*(?:руб\b|₽))", re.IGNORECASE,
)


def _quantities(question: str) -> set[int]:
    values = _QTY_UNIT.findall(question) or _QTY_LABEL.findall(question)
    return {int(value) if value.isdecimal() else _SMALL_NUMBERS[value.casefold()] for value in values}


def _numbers(text: str) -> dict[str, set[Decimal]]:
    found: dict[str, set[Decimal]] = {"price_rub": set(), "qty": set()}
    for amount, unit in _MEASURE.findall(text):
        field = "price_rub" if unit.casefold().startswith(("руб", "₽")) else "qty"
        found[field].add(Decimal("".join(amount.split()).replace(",", ".").replace("−", "-")))
    for label, amount in _LABELED.findall(text):
        field = "qty" if label.casefold() == "остаток" else "price_rub"
        found[field].add(Decimal("".join(amount.split()).replace(",", ".").replace("−", "-")))
    return found


def _unsupported(answer: str, question: str, steps: list[ToolStep]) -> bool:
    codes = set(find_codes(question))
    orders: set[str] = set()
    numbers = _numbers(question)
    for step in steps:
        if step.error or not step.result or step.result.get("refused"):
            continue
        codes.update(find_codes(json.dumps(step.result, ensure_ascii=False)))
        if step.result.get("order_id"):
            orders.add(str(step.result["order_id"]).upper())
        for field in numbers:
            value = step.result.get(field)
            if type(value) in (int, float):
                numbers[field].add(Decimal(str(value)))
    return (
        not set(find_codes(answer)) <= codes
        or not {value.upper() for value in _ORDER.findall(answer)} <= orders
        or any(not values <= numbers[field] for field, values in _numbers(answer).items())
    )


def _execute(tool: Callable[..., dict], args: dict) -> dict:
    for _ in range(3):
        try:
            return tool(**args)
        except ErpUnavailable:
            continue
    return {"error": _ERP_UNAVAILABLE, "kind": "unavailable"}


def _known_facts(result: AgentResult) -> list[str]:
    facts: dict[str, None] = {}
    for step in result.steps:
        data = step.result or {}
        if step.error:
            if data.get("kind") in {"unavailable", "not_found", "rejected"}:
                facts[step.error] = None
        elif data.get("order_id"):
            facts[f'Создан заказ {data["order_id"]}.'] = None
        elif "price_rub" in data:
            facts[f'{data["part_id"]}: {data["price_rub"]} руб.'] = None
        elif "qty" in data:
            facts[f'{data["part_id"]}, склад {data["warehouse"]}: {data["qty"]} шт.'] = None
        elif data.get("answer") and not data.get("refused"):
            facts[data["answer"]] = None
    return list(facts)


def _finish(result: AgentResult, answer: str) -> AgentResult:
    answer = answer.strip() or REFUSAL
    observations = [step.result for step in result.steps if step.result]
    has_data = any(not obs.get("refused") and "error" not in obs for obs in observations)
    missing = any(obs.get("refused") or obs.get("kind") == "not_found" for obs in observations)
    result.refused = REFUSAL.rstrip(".").casefold() in answer.casefold() or (missing and not has_data)
    if result.refused and REFUSAL.rstrip(".").casefold() not in answer.casefold():
        answer = f"{REFUSAL} {answer}"
    for step in result.steps:
        data = step.result or {}
        order = data.get("order_id")
        if order and order not in answer:
            answer += f" Создан заказ {order}."
        if data.get("kind") == "unavailable" and _ERP_UNAVAILABLE.casefold() not in answer.casefold():
            answer += f" {_ERP_UNAVAILABLE}"
        if data.get("kind") == "not_found" and step.error and step.error.casefold() not in answer.casefold():
            answer += f" {step.error}"
    result.answer = answer
    return result


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    result = AgentResult("")
    order_requested = bool(_ORDER_REQUEST.search(question))
    system = _SYSTEM + _ORDER_INSTRUCTION if order_requested else _SYSTEM
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    parts = set(find_codes(question))
    quantities = _quantities(question)
    completed: dict[tuple[str, str], dict] = {}
    order_retry_used = False
    schemas = TOOL_SCHEMAS

    while len(result.steps) < max_steps:
        reply = llm.chat(messages, tools=schemas, max_tokens=512)
        schemas = TOOL_SCHEMAS
        if not reply.tool_calls:
            if (
                order_requested and not order_retry_used and quantities and "create_service_order" in tools
                and any(code.startswith("P-") for code in parts)
                and not any(step.tool == "create_service_order" for step in result.steps)
            ):
                order_retry_used = True
                schemas = _ORDER_SCHEMA
                messages.append({
                    "role": "user",
                    "content": (
                        "Подготовь запрошенный заказ вызовом create_service_order, используя подтверждённые данные."
                    ),
                })
                continue
            answer = reply.content
            if _unsupported(answer, question, result.steps):
                messages.extend([
                    reply.as_message(),
                    {
                        "role": "user",
                        "content": (
                            "В ответе есть неподтверждённые коды, цены или остатки. Исправь ответ по вопросу "
                            "и результатам инструментов. Если данных нет, напиши «В базе знаний не найдено»."
                        ),
                    },
                ])
                answer = llm.chat(messages, max_tokens=384).content
                if _unsupported(answer, question, result.steps):
                    answer = REFUSAL
            return _finish(result, answer)

        messages.append(reply.as_message())
        remaining = max_steps - len(result.steps)
        for call in reply.tool_calls[:remaining]:
            name, args = call.name, call.arguments
            step = ToolStep(name, args if isinstance(args, dict) else {})
            result.steps.append(step)
            error = argument_error(name, args) if name in tools else "Неизвестный инструмент."
            if not error and "part_id" in args and args["part_id"] not in parts:
                error = "Номер детали не подтверждён. Сначала вызови search_knowledge."
            if not error and name == "create_service_order":
                if args["qty"] not in quantities:
                    step.error = "Количество не подтверждено пользователем."
                    result.answer = "Сколько штук нужно заказать?"
                    return result
                if not confirm:
                    step.error = "Требуется подтверждение."
                    result.needs_confirmation = True
                    result.pending_action = {"tool": name, "args": args}
                    result.answer = (
                        f'Подтвердите заказ детали {args["part_id"]} для {args["model"]}, '
                        f'{args["qty"]} шт., склад {args["warehouse"]}.'
                    )
                    return result
            if error:
                data = {"error": error}
            else:
                key = (name, json.dumps(args, ensure_ascii=False, sort_keys=True))
                if key not in completed:
                    completed[key] = _execute(tools[name], args)
                data = completed[key]
            step.result = data
            step.error = data.get("error")
            if not step.error and not data.get("refused"):
                parts.update(find_codes(json.dumps(data, ensure_ascii=False)))
            result.sources = list(dict.fromkeys([*result.sources, *data.get("sources", [])]))
            visible = {key: value for key, value in data.items() if key not in {"facts", "sources"}}
            messages.append(tool_message(call, visible))

    known = " ".join(_known_facts(result))
    return _finish(result, f"{known} Работа прервана: достигнут лимит вызовов инструментов.".strip())
