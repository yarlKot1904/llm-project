"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

from collections import defaultdict

import networkx as nx

from assistant.infra.schemas import Entity, ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    nodes: dict[Entity, set[str]] = defaultdict(set)
    edges: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for result in results:
        for entity in result.entities:
            nodes[entity].add(result.chunk_id)
        for relation in result.relations:
            if not relation.matches_schema():
                continue
            nodes[relation.subject].add(result.chunk_id)
            nodes[relation.object].add(result.chunk_id)
            key = (relation.subject.node_id, relation.object.node_id, relation.type)
            edges[key].add(result.chunk_id)

    graph = nx.DiGraph(conflicts=0)
    for entity, sources in sorted(nodes.items(), key=lambda item: item[0].node_id):
        graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=sorted(sources))
    for (source, target, kind), sources in sorted(edges.items()):
        if kind == "REPLACED_BY":
            if source == target:
                continue
            opposite = edges.get((target, source, kind))
            if opposite is not None:
                if source < target:
                    graph.graph["conflicts"] += 1
                if len(sources) <= len(opposite):
                    continue
        graph.add_edge(source, target, type=kind, weight=len(sources), sources=sorted(sources))
    return graph
