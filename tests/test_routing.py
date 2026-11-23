import numpy as np
from scipy.sparse import csr_matrix

from gravity_gnn.routing import k_nearest_network_sources


def line_graph(lengths: list[float]) -> csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for node, length in enumerate(lengths):
        rows.extend((node, node + 1))
        columns.extend((node + 1, node))
        values.extend((length, length))
    return csr_matrix(
        (values, (rows, columns)), shape=(len(lengths) + 1, len(lengths) + 1)
    )


def test_k_nearest_sources_are_sorted_and_include_snap_cost() -> None:
    graph = line_graph([1.0, 1.0, 1.0, 1.0])
    distance, source = k_nearest_network_sources(
        graph,
        source_nodes=np.array([0, 4]),
        source_ids=np.array([10, 20]),
        source_initial_cost=np.array([0.5, 0.0]),
        k=2,
        limit=10.0,
    )
    np.testing.assert_allclose(distance[2], [2.0, 2.5])
    np.testing.assert_array_equal(source[2], [20, 10])


def test_k_nearest_sources_respect_limit_and_disconnected_nodes() -> None:
    graph = csr_matrix(
        (
            [1.0, 1.0],
            ([0, 1], [1, 0]),
        ),
        shape=(3, 3),
    )
    distance, source = k_nearest_network_sources(
        graph,
        source_nodes=np.array([0]),
        source_ids=np.array([7]),
        source_initial_cost=np.array([0.0]),
        k=2,
        limit=0.5,
    )
    assert source[0, 0] == 7
    assert np.isinf(distance[1:]).all()
    assert (source[1:] == -1).all()
