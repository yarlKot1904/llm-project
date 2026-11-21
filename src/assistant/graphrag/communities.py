"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`."""
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
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще."""
    members = set(community.nodes)
    lines: list[str] = []
    family_failures: dict[str, Counter[str]] = {}
    family_models: dict[str, Counter[str]] = {}
    family_obsolete: dict[str, set[str]] = {}

    for model in sorted(node for node in members if graph.nodes[node].get("type") == "Model"):
        model_name = graph.nodes[model]["name"]
        outgoing = list(graph.out_edges(model, data=True))
        families = [graph.nodes[end]["name"] for _, end, attrs in outgoing if attrs["type"] == "BELONGS_TO"]
        family = families[0] if families else "Без линейки"
        failures: Counter[str] = Counter()
        obsolete: list[str] = []

        for _, end, attrs in outgoing:
            if attrs["type"] == "FAILED":
                comp_name = graph.nodes[end]["name"]
                failures[comp_name] += attrs["weight"]
                family_failures.setdefault(family, Counter())[comp_name] += attrs["weight"]
            if attrs["type"] == "HAS_PART" and any(
                edge["type"] == "REPLACED_BY" for _, _, edge in graph.out_edges(end, data=True)
            ):
                part_name = graph.nodes[end]["name"]
                obsolete.append(part_name)
                family_obsolete.setdefault(family, set()).add(part_name)

        total_fail = failures.total()
        family_models.setdefault(family, Counter())[model_name] = total_fail
        line = f"{model_name} (линейка «{family}»): всего отказов {total_fail}"
        if failures:
            breakdown = ", ".join(f"{node} — {count}" for node, count in failures.most_common())
            line += f" (по узлам: {breakdown})"
        if obsolete:
            line += f"; снятые детали из руководства: {', '.join(sorted(obsolete))}"
        lines.append(line)

    family_lines: list[str] = []
    for family in sorted(family_models):
        top_comps = ", ".join(f"{c} ({n})" for c, n in family_failures.get(family, Counter()).most_common())
        top_mods = ", ".join(f"{m} ({n})" for m, n in family_models[family].most_common())
        obs_list = ", ".join(sorted(family_obsolete.get(family, set()))) or "нет"
        family_lines.append(
            f"ИТОГО по линейке «{family}»: отказы по узлам: {top_comps}; "
            f"отказы по моделям: {top_mods}; все снятые с производства детали линейки: {obs_list}."
        )

    facts = [
        f'{graph.nodes[left]["name"]} -{attrs["type"]}-> {graph.nodes[right]["name"]} (вес {attrs["weight"]})'
        for left, right, attrs in sorted(graph.edges(data=True))
        if left in members and right in members
    ]
    nodes = ", ".join(graph.nodes[node]["name"] for node in community.nodes)
    evidence = "\n".join(facts[:60])

    if lines:
        context = "Сводка по линейкам:\n" + "\n".join(family_lines) + "\nПо моделям:\n" + "\n".join(lines)
        sys_prompt = (
            "Напиши по-русски сводку сообщества графа до 140 слов. "
            "Обязательно дословно сохрани итоговую статистику по линейке: какой узел отказывает чаще всего "
            "(с числами), сколько отказов у каждой модели (с числами) и полный список всех снятых с производства "
            "деталей из руководств этой линейки. Не меняй числа и коды."
        )
        user_prompt = f"{context}\nФакты:\n{evidence}"
    else:
        sys_prompt = (
            "Напиши по-русски краткую сводку сообщества графа до 100 слов: какие узлы, детали (P-...), "
            "ошибки (E-...), регламенты (SP-...) и поставщики входят в группу и как они связаны. "
            "В этом сообществе НЕТ моделей оборудования и НЕТ статистики отказов — не упоминай отказы."
        )
        user_prompt = f"Сущности: {nodes}\nФакты:\n{evidence}"

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    reply = llm.chat(messages, max_tokens=512)
    return " ".join(reply.content.split()[:150])