"""Block E - controlled damage to the graph (random edge removal / addition)."""


def remove_edges(g, fraction, rng):
    """Copy of g with a random `fraction` of the edges deleted. Nodes stay."""
    raise NotImplementedError("step 6")


def add_edges(g, fraction, rng):
    """Copy of g with `fraction * |E|` random new edges. Nodes stay."""
    raise NotImplementedError("step 6")
