"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    projection = nx.Graph()
    projection.add_nodes_from(sorted(graph))
    for left, right, attrs in sorted(graph.edges(data=True)):
        weight = attrs.get("weight", 1) + projection.get_edge_data(left, right, {}).get("weight", 0)
        projection.add_edge(left, right, weight=weight)
    if not projection.number_of_edges():
        return [Community(i, [node]) for i, node in enumerate(projection)]
    groups = nx.community.louvain_communities(projection, weight="weight", seed=42)
    return [Community(i, nodes) for i, nodes in enumerate(sorted(sorted(group) for group in groups))]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    members = set(community.nodes)
    lines: list[str] = []
    for model in sorted(node for node in members if graph.nodes[node].get("type") == "Model"):
        outgoing = list(graph.out_edges(model, data=True))
        families = [graph.nodes[end]["name"] for _, end, attrs in outgoing if attrs["type"] == "BELONGS_TO"]
        failures = Counter()
        obsolete = []
        for _, end, attrs in outgoing:
            if attrs["type"] == "FAILED":
                failures[graph.nodes[end]["name"]] += attrs["weight"]
            if attrs["type"] == "HAS_PART" and any(
                edge["type"] == "REPLACED_BY" for _, _, edge in graph.out_edges(end, data=True)
            ):
                obsolete.append(graph.nodes[end]["name"])
        line = f'{graph.nodes[model]["name"]} ({", ".join(families) or "линейка не указана"})'
        if failures:
            breakdown = ", ".join(f"{node}: {count}" for node, count in failures.most_common())
            line += f": всего отказов {failures.total()}, по узлам {breakdown}"
        if obsolete:
            line += f"; снятые детали из руководства: {', '.join(sorted(obsolete))}"
        lines.append(line)

    facts = [
        f'{graph.nodes[left]["name"]} -{attrs["type"]}-> {graph.nodes[right]["name"]} (вес {attrs["weight"]})'
        for left, right, attrs in sorted(graph.edges(data=True))
        if left in members and right in members
    ]
    nodes = ", ".join(graph.nodes[node]["name"] for node in community.nodes)
    context = "\n".join(lines)
    evidence = "\n".join(facts[:60])
    messages = [
        {
            "role": "system",
            "content": (
                "Напиши по-русски сводку сообщества графа до 150 слов. "
                "Перенеси готовые сведения по моделям: линейки, распределение отказов по узлам, "
                "число отказов каждой модели и снятые детали из её руководства. "
                "Числа уже рассчитаны по отчётам, не пересчитывай их и не добавляй фактов. "
                "Если место осталось, кратко упомяни другие связи сообщества."
            ),
        },
        {"role": "user", "content": f"Модели:\n{context}\nСущности: {nodes}\nФакты:\n{evidence}"},
    ]
    reply = llm.chat(messages, max_tokens=512)
    return " ".join(reply.content.split()[:150])
