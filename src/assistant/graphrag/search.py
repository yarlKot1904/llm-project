"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations, pairwise

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

REFUSAL = "В базе знаний не найдено."
_MAX_FACT_CHARS = 8000
_LOCAL_SYSTEM = (
    "Ты ассистент сервисной службы. Сразу назови запрошенный результат по фактам, одним или двумя предложениями. "
    "INDICATES указывает вид узла, HAS_PART связывает модель с номером детали, IS_A связывает деталь с видом узла. "
    "FIXED_BY указывает регламент для кода ошибки, BELONGS_TO — линейку модели, SUPPLIED_BY — поставщика детали. "
    "FAILED означает зафиксированный отказ узла. "
    "Для детали в модели сопоставь HAS_PART и IS_A с нужным видом узла. "
    "REPLACED_BY заменяет номер детали, а не модель. "
    "Если нужна актуальная деталь, пройди от старой до последней новой. "
    "Укажи поставщика актуальной детали, только если спрашивают о поставщике. "
    "При вопросе об установленных деталях указывай модели с нужной HAS_PART. "
    "Сохраняй коды без изменений. Если факты не отвечают на вопрос, напиши «В базе знаний не найдено»."
)


def _sources(graph: nx.DiGraph, edges: list[tuple[str, str]]) -> list[str]:
    return list(dict.fromkeys(
        source.split("#", 1)[0] for edge in edges for source in graph.edges[edge].get("sources", [])
    ))


def _anchors(question: str, kb: KnowledgeBase) -> list[str] | None:
    codes = [f"{code_type(code)}:{code}" for code in find_codes(question)]
    if any(node not in kb.graph for node in codes):
        return None
    names = [f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary)]
    return list(dict.fromkeys(node for node in [*codes, *names] if node in kb.graph))


def _paths(graph: nx.DiGraph, anchors: list[str]) -> dict[tuple[str, str], None]:
    # The view merges neighbours through unordered sets; a copy preserves insertion order.
    undirected = graph.to_undirected()
    if len(anchors) == 1:
        distances = nx.single_source_shortest_path_length(undirected, anchors[0], cutoff=2)
        return dict.fromkeys(sorted(
            graph.subgraph(distances).edges(),
            key=lambda edge: (max(distances[edge[0]], distances[edge[1]]), edge),
        ))

    edges: dict[tuple[str, str], None] = {}
    for left, right in combinations(anchors, 2):
        for path in nx.all_simple_paths(undirected, left, right, cutoff=3):
            for start, end in pairwise(path):
                if graph.has_edge(start, end):
                    edges[start, end] = None
                if graph.has_edge(end, start):
                    edges[end, start] = None
    return edges


def _add_replacements(graph: nx.DiGraph, edges: dict[tuple[str, str], None], suppliers: bool) -> None:
    pending = list(dict.fromkeys(node for edge in edges for node in edge if node.startswith("Part:")))
    seen = set(pending)
    for part in pending:
        replacement = False
        for _, target, attrs in graph.out_edges(part, data=True):
            if attrs["type"] != "REPLACED_BY":
                continue
            replacement = True
            edges[part, target] = None
            if target not in seen:
                seen.add(target)
                pending.append(target)
        if suppliers and not replacement:
            for _, target, attrs in graph.out_edges(part, data=True):
                if attrs["type"] == "SUPPLIED_BY":
                    edges[part, target] = None


def _answered(text: str, sources: list[str], facts: list[str] | None = None) -> Answer:
    if not text.strip() or "в базе знаний не найдено" in text.casefold():
        return Answer(REFUSAL, sources=sources, facts=facts or [], refused=True)
    return Answer(text.strip(), sources=sources, facts=facts or [])


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    anchors = _anchors(question, kb)
    if not anchors:
        return Answer(REFUSAL, refused=True)

    graph = kb.graph
    edges = _paths(graph, anchors)
    _add_replacements(graph, edges, suppliers="постав" in question.casefold())
    facts: list[str] = []
    selected: list[tuple[str, str]] = []
    size = 0
    for left, right in edges:
        kind = graph.edges[left, right]["type"]
        fact = f'{graph.nodes[left]["name"]} -{kind}-> {graph.nodes[right]["name"]}'
        line = f"{len(facts) + 1}. {fact}"
        if size + len(line) + 1 > _MAX_FACT_CHARS:
            break
        facts.append(fact)
        selected.append((left, right))
        size += len(line) + 1
    if not facts:
        return Answer(REFUSAL, refused=True)

    context = "\n".join(f"{i}. {fact}" for i, fact in enumerate(facts, 1))
    messages = [
        {"role": "system", "content": _LOCAL_SYSTEM},
        {"role": "user", "content": f"Факты:\n{context}\n\nВопрос: {question}"},
    ]
    reply = llm.chat(messages, max_tokens=384)
    return _answered(reply.content, _sources(graph, selected), facts)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    if _anchors(question, kb) is None:
        return Answer(REFUSAL, refused=True)

    parts: list[tuple[int, str]] = []
    sources: dict[str, None] = {}
    for community in kb.communities:
        if not community.summary.strip():
            continue
        messages = [
            {
                "role": "system",
                "content": (
                    "Оцени, что эта сводка даёт для ответа на вопрос. Верни JSON с answer и score от 0 до 10. "
                    'При нерелевантной сводке верни {"answer":"","score":0}. '
                    "Иначе выпиши относящиеся к вопросу модели, детали и численные данные; "
                    "сохрани разбивку по моделям, если она есть. "
                    "Не дополняй отсутствующие данные."
                ),
            },
            {"role": "user", "content": f"Сводка:\n{community.summary}\n\nВопрос: {question}"},
        ]
        reply = llm.chat(messages, json_schema=MAP_SCHEMA, max_tokens=384)
        try:
            partial = json.loads(reply.content)
        except ValueError:
            continue
        if not isinstance(partial, dict):
            continue
        text, score = partial.get("answer"), partial.get("score")
        if not isinstance(text, str) or not text.strip() or type(score) is not int or not 0 < score <= 10:
            continue
        parts.append((score, text.strip()))
        for node in community.nodes:
            if node in kb.graph:
                for source in kb.graph.nodes[node].get("sources", []):
                    sources[source.split("#", 1)[0]] = None
    if not parts:
        return Answer(REFUSAL, refused=True)

    context = "\n".join(f"- {text}" for _, text in sorted(parts, key=lambda part: -part[0]))
    messages = [
        {
            "role": "system",
            "content": (
                "Ответь по частичным ответам сообществ. Учитывай указанную в вопросе линейку. "
                "При сравнении частоты учитывай числа из всех релевантных частей, не дублируй модели. "
                "Кратко назови результат и поясни его. Если ответа нет, напиши «В базе знаний не найдено»."
            ),
        },
        {"role": "user", "content": f"Вопрос: {question}\nЧастичные ответы:\n{context}"},
    ]
    return _answered(llm.chat(messages, max_tokens=384).content, list(sources))
