"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json
import re

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, code_type, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

_SYSTEM = """Извлеки факты только из данного фрагмента сервисной документации.
Связи направленные, используй ровно следующие типы и направления:
Model -BELONGS_TO-> ProductLine: модель входит в линейку.
Model -HAS_PART-> Part: деталь установлена в модели (строка таблицы состава).
Part -IS_A-> Component: вид узла, которому соответствует номер детали.
ErrorCode -INDICATES-> Component: код ошибки указывает на вид узла.
ErrorCode -FIXED_BY-> Procedure: ошибку устраняют по этому регламенту.
Part -REPLACED_BY-> Part: старая деталь заменена новой. Только от старой к новой.
Part -SUPPLIED_BY-> Supplier: поставщик конкретного номера детали.
Model -FAILED-> Component: фактический отказ из отчёта о выезде, не таблица возможных ошибок.

В таблицах извлекай каждую строку отдельно. Код модели из заголовка относится к таблице.
В отчёте извлекай FAILED по фактическому диагнозу, также INDICATES если указан код ошибки.
В бюллетене не переноси поставщика новой детали на старую. Дефект партии не означает замену.
В регламенте связывай код ошибки с номером регламента, а не с кодом детали.
Если описаны только плановые работы без связи с ошибками, верни пустые списки.
Сохраняй обозначения и имена из фрагмента. Не добавляй связь из общих знаний.
Формат: компактный JSON с массивами entities и relations. Достаточно одного экземпляра каждого факта.
В entities указывай только сущности, которые не записаны в relations. Концы связей восстанавливает код.
Например для «E-31 означает отказ узла контроллер»:
{"entities":[],"relations":[{"subject_type":"ErrorCode","subject":"E-31","type":"INDICATES",
"object_type":"Component","object":"контроллер"}]}"""

_ENTITY_FIELDS = ("type", "name")
_RELATION_FIELDS = ("subject_type", "subject", "type", "object_type", "object")
_PART_ROW = re.compile(r"^- (.+?)\s*[–—-]\s*деталь\s+(\S+),\s*поставщик\s+«([^»]+)»\.?$")
_ERROR_ROW = re.compile(r"^- (\S+)\s+[–—-]\s+.+?,\s*отказ узла\s+«([^»]+)»\.?$")
_EQUIPMENT = re.compile(r"^Оборудование: ([^.\n]+)\. На панели горел код ([^,.\n]+), [^.\n]+\.$", re.M)
_DIAGNOSIS = re.compile(r"^Диагностика показала отказ узла: ([^.\n]+)\.[ \t]*", re.M)
_APPLIES = re.compile(r"^Регламент (\S+) применяется при ошибке (\S+)\.[ \t]*", re.M)


def _row(left_type: str, left: str, kind: str, right_type: str, right: str) -> dict[str, str]:
    return dict(zip(_RELATION_FIELDS, (left_type, left, kind, right_type, right), strict=True))


def _manual_rows(text: str, names: dict[str, list[str]]) -> tuple[str, list[dict[str, str]]]:
    """Разбирает явные строки руководства, оставляя нестандартные записи для модели."""
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# Руководство"):
        return text, []
    models = [code for code in find_codes(lines[0]) if code_type(code) == "Model"]
    if len(models) != 1:
        return text, []

    remaining = []
    rows = []
    for line in lines:
        part_match = _PART_ROW.fullmatch(line)
        if part_match:
            component, part, supplier = part_match.groups()
            if (
                canonical_name("Part", part, names)
                and canonical_name("Component", component, names)
                and canonical_name("Supplier", supplier, names)
            ):
                rows.extend([
                    _row("Model", models[0], "HAS_PART", "Part", part),
                    _row("Part", part, "IS_A", "Component", component),
                    _row("Part", part, "SUPPLIED_BY", "Supplier", supplier),
                ])
                continue
        error_match = _ERROR_ROW.fullmatch(line)
        if error_match:
            error, component = error_match.groups()
            if canonical_name("ErrorCode", error, names) and canonical_name("Component", component, names):
                rows.append(_row("ErrorCode", error, "INDICATES", "Component", component))
                continue
        remaining.append(line)
    return "\n".join(remaining) if rows else text, rows


def _structured_rows(text: str, names: dict[str, list[str]]) -> tuple[str, list[dict[str, str]]]:
    if text.startswith("# Отчёт о выезде"):
        equipment, diagnosis = _EQUIPMENT.search(text), _DIAGNOSIS.search(text)
        if equipment and diagnosis:
            models = [code for code in find_codes(equipment[1]) if code_type(code) == "Model"]
            error = canonical_name("ErrorCode", equipment[2], names)
            component = canonical_name("Component", diagnosis[1], names)
            if len(models) == 1 and error and component:
                prose = _DIAGNOSIS.sub("", text, count=1)
                prose = _EQUIPMENT.sub("", prose, count=1)
                return prose, [
                    _row("Model", models[0], "FAILED", "Component", component),
                    _row("ErrorCode", error, "INDICATES", "Component", component),
                ]
    if text.startswith("# Регламент"):
        applies = _APPLIES.search(text)
        if applies:
            procedure = canonical_name("Procedure", applies[1], names)
            error = canonical_name("ErrorCode", applies[2], names)
            if procedure and error:
                description = re.compile(
                    rf"Ошибка {re.escape(applies[2])} указывает на отказ узла «([^»]+)»\.[ \t]*"
                )
                row = description.match(text, applies.end())
                component = canonical_name("Component", row[1], names) if row else None
                if row and component:
                    return text[:applies.start()] + text[row.end():], [
                        _row("ErrorCode", error, "FIXED_BY", "Procedure", procedure),
                        _row("ErrorCode", error, "INDICATES", "Component", component),
                    ]
    return _manual_rows(text, names)


def _read_rows(data: dict, key: str, fields: tuple[str, ...]) -> list[dict[str, str]]:
    rows = data.get(key)
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or any(not isinstance(row.get(field), str) for field in fields)
        for row in rows
    ):
        raise ValueError(f"Некорректный массив {key}")
    return rows


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает проверенные сущности и связи, сохраняя идентификаторы источника."""
    result = ExtractionResult(chunk.chunk_id, chunk.doc_id)
    names = vocabulary()
    prose, explicit_rows = _structured_rows(chunk.text, names)
    user_text = f"Фрагмент документа:\n{prose}"
    if explicit_rows:
        if chunk.text.startswith("# Руководство"):
            user_text += "\n\nСтроки состава и кодов ошибок уже обработаны; извлеки только факты из оставшегося текста."
        else:
            user_text += (
                "\n\nЯвные поля ошибок и отказов уже обработаны. Извлеки только новые факты из оставшегося текста. "
                'Если таких фактов нет, верни {"entities":[],"relations":[]}.'
            )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_text},
    ]
    for _ in range(2):
        reply = llm.chat(messages, json_schema=EXTRACTION_SCHEMA, max_tokens=2304)
        try:
            data = json.loads(reply.content)
            if not isinstance(data, dict):
                raise ValueError("Ожидался объект JSON")
            entities = _read_rows(data, "entities", _ENTITY_FIELDS)
            relations = _read_rows(data, "relations", _RELATION_FIELDS)
            break
        except ValueError:
            messages = [
                *messages[:2],
                {"role": "user", "content": "Верни только полный JSON со всеми обязательными полями схемы."},
            ]
    else:
        return result

    codes = set(find_codes(chunk.text))

    def entity(kind: str, raw: str) -> Entity | None:
        name = canonical_name(kind, raw, names)
        if kind not in ENTITY_TYPES or not name or (kind in CODE_TYPES and name not in codes):
            return None
        return Entity(kind, name)

    unique_entities: dict[Entity, None] = {}
    unique_relations: dict[Relation, None] = {}
    for row in entities:
        node = entity(row["type"], row["name"])
        if node is None:
            result.dropped += 1
        else:
            unique_entities[node] = None

    for row in [*explicit_rows, *relations]:
        subject = entity(row["subject_type"], row["subject"])
        target = entity(row["object_type"], row["object"])
        if subject is None or target is None or RELATION_SIGNATURES.get(row["type"]) != (subject.type, target.type):
            result.dropped += 1
            continue
        unique_entities[subject] = None
        unique_entities[target] = None
        unique_relations[Relation(subject, row["type"], target)] = None

    result.entities = list(unique_entities)
    result.relations = list(unique_relations)
    return result
