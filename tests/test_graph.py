"""Dynamic graph construction and GraphState behaviour."""

from sentinelx.graph.types import EdgeState, GraphState, NodeState


def test_sequence_length_matches_windows(graph_sequence):
    assert len(graph_sequence) == 40
    assert all(isinstance(g, GraphState) for g in graph_sequence)


def test_nodes_and_edges_present(graph_sequence):
    g = graph_sequence[0]
    assert g.node_count() > 0
    assert g.edge_count() > 0
    assert len(g.node_feature_names) == 8


def test_servers_detected(graph_sequence):
    servers = graph_sequence[0].server_keys()
    assert any(s.startswith("SERVER") for s in servers)


def test_edges_have_dst_port(graph_sequence):
    for e in graph_sequence[0].edges:
        assert isinstance(e.dst_port, int)


def test_embedding_is_fixed_length(graph_sequence):
    dims = {len(g.embedding()) for g in graph_sequence if g.node_count() > 0}
    assert len(dims) == 1  # same length regardless of node count


def test_clone_is_deep():
    g = GraphState(index=0, window_start=0, window_end=60, node_feature_names=["a"])
    g.nodes["n1"] = NodeState("n1", "n1", [1.0])
    g.edges.append(EdgeState("n1", "n2", "TCP", [1.0], 1.0, 80))
    c = g.clone()
    c.nodes["n1"].features[0] = 99.0
    assert g.nodes["n1"].features[0] == 1.0  # original untouched


def test_adjacency_and_edge_set():
    g = GraphState(index=0, window_start=0, window_end=60)
    g.nodes["a"] = NodeState("a", "a", [])
    g.nodes["b"] = NodeState("b", "b", [])
    g.edges.append(EdgeState("a", "b", "TCP", [], 1.0))
    assert g.adjacency()["a"] == ["b"]
    assert ("a", "b") in g.edge_set()


def test_to_json_roundtrip_shape(graph_sequence):
    j = graph_sequence[5].to_json()
    assert "nodes" in j and "edges" in j
    assert all("features" in n for n in j["nodes"])
