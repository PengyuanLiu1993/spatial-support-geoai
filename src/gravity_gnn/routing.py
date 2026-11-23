from __future__ import annotations

import heapq

import numpy as np
from scipy.sparse import csr_matrix


def k_nearest_network_sources(
    graph: csr_matrix,
    source_nodes: np.ndarray,
    source_ids: np.ndarray,
    source_initial_cost: np.ndarray,
    k: int,
    limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the ``k`` nearest distinct sources to every graph node.

    This is a labelled multi-source Dijkstra search.  Each source propagates a
    label through the network, while a node stops forwarding new labels after
    its first ``k`` distinct labels have been accepted.  Initial costs account
    for a POI's snap distance to the road network.

    Returns
    -------
    distances, ids:
        Arrays with shape ``[num_nodes, k]`` sorted by network cost. Missing
        neighbours are represented by ``inf`` and ``-1``.
    """

    if not isinstance(graph, csr_matrix):
        graph = csr_matrix(graph)
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("graph must be square")
    if k < 1 or limit <= 0:
        raise ValueError("k and limit must be positive")
    source_nodes = np.asarray(source_nodes, dtype=np.int64)
    source_ids = np.asarray(source_ids, dtype=np.int64)
    source_initial_cost = np.asarray(source_initial_cost, dtype=np.float64)
    if not (
        source_nodes.shape == source_ids.shape == source_initial_cost.shape
        and source_nodes.ndim == 1
    ):
        raise ValueError("source arrays must be one-dimensional and aligned")
    if len(source_nodes) == 0:
        return (
            np.full((graph.shape[0], k), np.inf, dtype=np.float64),
            np.full((graph.shape[0], k), -1, dtype=np.int64),
        )
    if source_nodes.min() < 0 or source_nodes.max() >= graph.shape[0]:
        raise ValueError("source node is outside the graph")
    if not np.isfinite(source_initial_cost).all() or np.any(source_initial_cost < 0):
        raise ValueError("source initial costs must be finite and non-negative")
    if len(np.unique(source_ids)) != len(source_ids):
        raise ValueError("source ids must be unique")

    distances = np.full((graph.shape[0], k), np.inf, dtype=np.float64)
    ids = np.full((graph.shape[0], k), -1, dtype=np.int64)
    accepted_count = np.zeros(graph.shape[0], dtype=np.int16)
    queue: list[tuple[float, int, int]] = []
    for node, source_id, initial_cost in zip(
        source_nodes, source_ids, source_initial_cost, strict=True
    ):
        if initial_cost <= limit:
            heapq.heappush(queue, (float(initial_cost), int(node), int(source_id)))

    indptr = graph.indptr
    indices = graph.indices
    weights = graph.data
    while queue:
        distance, node, source_id = heapq.heappop(queue)
        if distance > limit:
            break
        count = int(accepted_count[node])
        if count and source_id in ids[node, :count]:
            continue
        if count >= k:
            continue

        distances[node, count] = distance
        ids[node, count] = source_id
        accepted_count[node] = count + 1
        for edge_position in range(indptr[node], indptr[node + 1]):
            neighbour = int(indices[edge_position])
            if accepted_count[neighbour] >= k:
                continue
            candidate = distance + float(weights[edge_position])
            if candidate <= limit:
                heapq.heappush(queue, (candidate, neighbour, source_id))

    return distances, ids
