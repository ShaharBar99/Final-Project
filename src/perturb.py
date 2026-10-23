"""Block E - controlled damage to the graph (random edge removal / addition).

Three kinds of perturbation:
    remove      delete a random fraction of the existing edges (missing data)
    add_random  add random new edges between any two papers (Phase A as written)
    add_dag     add random new edges that respect the time order of the graph,
                so no citation cycle can appear (decision D10)

In all three the set of nodes never changes - only edges do.
"""
import networkx as nx
import numpy as np

KINDS = ["remove", "add_random", "add_dag"]


def remove_edges(g, fraction, rng):
    """Copy of g with a random `fraction` of the edges deleted. Nodes stay."""
    edges = list(g.edges)
    n_remove = int(round(fraction * len(edges)))
    picked = rng.choice(len(edges), size=n_remove, replace=False)

    damaged = g.copy()
    damaged.remove_edges_from(edges[i] for i in picked)
    return damaged


def time_order(g):
    """Position of every node in a fixed topological order of g.

    In a citation DAG an edge u -> v means u is newer than v, so a topological
    order goes from newest to oldest. Any order that agrees with all existing
    edges will do; the lexicographic one is used so the result never changes.
    """
    order = nx.lexicographical_topological_sort(g)
    return {node: i for i, node in enumerate(order)}


def add_edges(g, fraction, rng, order=None):
    """Copy of g with `fraction * |E|` new random edges. Nodes stay.

    A new edge is never a self-loop and never a copy of an edge that already
    exists. If `order` is given (see time_order), every new edge goes from the
    earlier node in that order to the later one, so the graph stays a DAG.
    """
    nodes = list(g.nodes)
    n_add = int(round(fraction * g.number_of_edges()))

    damaged = g.copy()
    added = 0
    while added < n_add:
        # draw in batches, it is much faster than one pair at a time
        pairs = rng.integers(0, len(nodes), size=(2 * (n_add - added) + 10, 2))
        for i, j in pairs:
            if i == j:
                continue
            u, v = nodes[i], nodes[j]
            if order is not None and order[u] > order[v]:
                u, v = v, u
            if damaged.has_edge(u, v):
                continue
            damaged.add_edge(u, v)
            added += 1
            if added == n_add:
                break
    return damaged


def perturb(g, kind, fraction, rng, order=None):
    """One perturbed copy of g. `order` is only used by add_dag."""
    if kind == "remove":
        return remove_edges(g, fraction, rng)
    if kind == "add_random":
        return add_edges(g, fraction, rng)
    if kind == "add_dag":
        return add_edges(g, fraction, rng, order=order if order is not None else time_order(g))
    raise ValueError(f"unknown perturbation kind: {kind}")


def run_rng(seed, kind, fraction, run):
    """Random generator for one single run.

    Built from (seed, kind, level, run number), so every run can be repeated
    on its own and changing one setting does not shift the others.
    """
    return np.random.default_rng([seed, KINDS.index(kind), int(round(fraction * 100)), run])
