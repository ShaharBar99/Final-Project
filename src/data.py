"""Loading and cleaning of the citation data (PubMed-Diabetes, Cora, CiteSeer).

Every cleaning step records what it did in a `log` list, so the cleaning
report can be built from what actually happened and not from what we remember.
"""
import json

import igraph as ig
import networkx as nx
import pandas as pd

from . import config


# ---------------------------------------------------------------- loading

def load_nodes(path=config.NODES_FILE):
    """Read the node file. Returns a DataFrame with columns: paper_id, label.

    The file has a header line, a line describing the feature columns, and then
    one line per paper. Each paper line looks like:
        <id>  label=<1|2|3>  w-xxx=0.05  ...  summary=...
    We only need the id and the label (the word features are not used).
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        next(f)  # "NODE paper"
        next(f)  # description of the feature columns
        for line in f:
            parts = line.rstrip("\n").split("\t")
            label = next(p for p in parts[1:] if p.startswith("label="))
            rows.append((parts[0], int(label.split("=")[1])))
    return pd.DataFrame(rows, columns=["paper_id", "label"])


def load_edges(path=config.EDGES_FILE):
    """Read the citation file. Returns a DataFrame with columns: source, target.

    A line like  `33824  paper:19127292 | paper:17363749`  means that
    paper 19127292 cites paper 17363749, so the edge goes source -> target.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        next(f)  # "DIRECTED cites"
        next(f)  # "NO_FEATURES"
        for line in f:
            parts = line.split()
            rows.append((parts[1].split(":")[1], parts[3].split(":")[1]))
    return pd.DataFrame(rows, columns=["source", "target"])


def load_linqs(name):
    """Read Cora or CiteSeer from data/raw/<name>/.

    <name>.nodes.tab   paper id and class label (made from the .content file, see data/raw/<name>/SOURCE.md)
    <name>.cites       lines "<cited paper> <citing paper>", i.e. the link goes from the SECOND id to the first.
    Returns (nodes, edges) in the same shape as load_nodes / load_edges: edge = source cites target.
    """
    folder = config.RAW_DIR / name
    nodes = pd.read_csv(folder / f"{name}.nodes.tab", sep="	", dtype=str)
    cites = pd.read_csv(folder / f"{name}.cites", sep="	", header=None, names=["cited", "citing"], dtype=str)
    edges = pd.DataFrame({"source": cites["citing"], "target": cites["cited"]})
    return nodes, edges


# ---------------------------------------------------------------- cleaning steps

def remove_duplicate_edges(edges, log):
    before = len(edges)
    edges = edges.drop_duplicates()
    log.append({"step": "remove_duplicate_edges",
                "removed": before - len(edges), "edges_left": len(edges)})
    return edges


def remove_self_loops(edges, log):
    """A paper citing itself carries no information about its importance."""
    is_loop = edges["source"] == edges["target"]
    log.append({"step": "remove_self_loops",
                "removed": int(is_loop.sum()), "edges_left": int((~is_loop).sum()),
                "examples": edges[is_loop]["source"].tolist()[:5]})
    return edges[~is_loop]


def resolve_mutual_citations(edges, log):
    """Break every pair of papers that cite each other (A -> B and B -> A).

    Two papers can't really cite each other, since the older one can't know
    the newer one. The data has no publication years, so we use the PubMed ID
    as a stand-in for time: IDs are handed out in increasing order, so a
    bigger ID usually means a newer paper. We keep the edge that goes from the
    newer paper (bigger ID) to the older one and drop the opposite edge.

    This is an assumption, not a fact - it is written down in docs/decisions.md.
    """
    pairs = set(zip(edges["source"], edges["target"]))
    to_drop = []
    for a, b in pairs:
        if (b, a) in pairs and int(a) < int(b):
            # a -> b goes from the older to the newer paper, so it is the wrong one
            to_drop.append((a, b))
    drop_index = edges.set_index(["source", "target"]).index.isin(to_drop)
    log.append({"step": "resolve_mutual_citations",
                "removed": len(to_drop), "edges_left": int((~drop_index).sum()),
                "examples": [f"{a}->{b}" for a, b in to_drop[:5]]})
    return edges[~drop_index]


def keep_known_papers(edges, nodes, log):
    """Drop edges that point to / come from a paper that is not in the node file."""
    known = set(nodes["paper_id"])
    ok = edges["source"].isin(known) & edges["target"].isin(known)
    log.append({"step": "drop_edges_with_unknown_papers",
                "removed": int((~ok).sum()), "edges_left": int(ok.sum())})
    return edges[ok]


def remove_isolated_nodes(nodes, edges, log):
    """Papers without any citation (in or out) cannot be ranked structurally."""
    connected = set(edges["source"]) | set(edges["target"])
    keep = nodes["paper_id"].isin(connected)
    log.append({"step": "remove_isolated_nodes",
                "removed": int((~keep).sum()), "nodes_left": int(keep.sum())})
    return nodes[keep]


def break_citation_cycles(edges, log):
    """Remove the fewest citations needed so that no citation loop is left.

    Papers cannot cite each other, directly or through a chain, because a paper
    can only cite older papers. PubMed IDs grow with time, so for PubMed we could
    drop the single wrong edge of each mutual pair (resolve_mutual_citations).
    Cora and CiteSeer have no dates and their ids are not in time order, so we
    cannot tell which edge of a loop is the wrong one. Instead we remove a
    minimum set of edges that makes the graph acyclic (a minimum feedback arc set,
    solved exactly for every strongly connected group). See docs/decisions.md D18.
    """
    papers = pd.unique(pd.concat([edges["source"], edges["target"]]))
    index = {paper: i for i, paper in enumerate(papers)}
    pairs = [(index[a], index[b]) for a, b in zip(edges["source"], edges["target"])]
    graph = ig.Graph(n=len(papers), edges=pairs, directed=True)
    pair_set = set(pairs)
    mutual = sum(1 for a, b in pair_set if a < b and (b, a) in pair_set)

    drop = set()
    groups = [g for g in graph.connected_components(mode="strong") if len(g) > 1]
    for group in groups:
        sub = graph.induced_subgraph(group)
        for edge_id in sub.feedback_arc_set(method="ip"):
            a, b = sub.es[edge_id].tuple
            drop.add((group[a], group[b]))
    removed = [pair in drop for pair in pairs]
    log.append({"step": "break_citation_cycles",
                "mutual_pairs_before": mutual, "loop_groups_before": len(groups),
                "removed": int(sum(removed)), "edges_left": int(len(pairs) - sum(removed))})
    return edges[[not r for r in removed]]


def check_graph(edges, log):
    """Final sanity checks - these only report, they do not change the data."""
    g = nx.from_pandas_edgelist(edges, "source", "target", create_using=nx.DiGraph)
    n_weak = nx.number_weakly_connected_components(g)
    n_cycles = sum(1 for c in nx.strongly_connected_components(g) if len(c) > 1)
    log.append({"step": "final_checks",
                "nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
                "weakly_connected_components": n_weak,
                "directed_cycles_groups": n_cycles})
    return g


# ---------------------------------------------------------------- full pipeline

def clean_dataset(dataset=None):
    """Run all cleaning steps in order. Returns (nodes, edges, log).

    PubMed-Diabetes keeps the steps of the original study (mutual citations are resolved with the
    PubMed-ID rule). Cora and CiteSeer use the same steps, but loops are removed by
    break_citation_cycles because they have no usable time information.
    """
    dataset = dataset or config.DATASET
    log = []
    if dataset == "pubmed":
        nodes, edges = load_nodes(), load_edges()
    else:
        nodes, edges = load_linqs(dataset)
    log.append({"step": "load_raw", "nodes": len(nodes), "edges": len(edges),
                "label_counts": nodes["label"].value_counts().sort_index().to_dict()})

    edges = remove_duplicate_edges(edges, log)
    edges = remove_self_loops(edges, log)
    if dataset == "pubmed":
        edges = resolve_mutual_citations(edges, log)
        edges = keep_known_papers(edges, nodes, log)
    else:
        edges = keep_known_papers(edges, nodes, log)
        edges = break_citation_cycles(edges, log)
    nodes = remove_isolated_nodes(nodes, edges, log)
    check_graph(edges, log)

    return nodes.reset_index(drop=True), edges.reset_index(drop=True), log


def save_dataset(nodes, edges, log):
    """Write the cleaned data and the machine-readable log to disk."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(config.PROCESSED_DIR / "nodes.csv", index=False)
    edges.to_csv(config.PROCESSED_DIR / "edges.csv", index=False)
    with open(config.PROCESSED_DIR / "cleaning_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def load_clean_graph():
    """Convenience for the next steps: the cleaned data as a networkx DiGraph."""
    nodes = pd.read_csv(config.PROCESSED_DIR / "nodes.csv", dtype={"paper_id": str})
    edges = pd.read_csv(config.PROCESSED_DIR / "edges.csv", dtype=str)
    g = nx.from_pandas_edgelist(edges, "source", "target", create_using=nx.DiGraph)
    g.add_nodes_from(nodes["paper_id"])
    return g, nodes
