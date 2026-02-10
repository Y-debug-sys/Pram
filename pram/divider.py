import os
import networkx as nx
import matplotlib.pyplot as plt

from collections import defaultdict


def draw_paths_per_source(
    G: nx.Graph,
    kpaths: dict,
    out_dir: str = "ksp_plots",
    dpi: int = 240,
    arrowstyle: str = "-|>",   # used if G is directed
    base_arrowsize: int = 12,
    scale: float = 1.0,
    plot_cap: bool = False,
    plot_arrow: bool = True
):
    """
    Draws all candidate paths grouped by source node.
    Each source gets one figure containing all its candidate paths (to all destinations).
    Parameters (node size, font, edge width) are auto-scaled to avoid overlaps.
    """

    os.makedirs(out_dir, exist_ok=True)
    is_directed = (G.is_directed() and plot_arrow)

    # Group paths by source
    grouped_paths = defaultdict(list)
    for (s, t), path_list in kpaths.items():
        for epath in path_list:
            grouped_paths[s].append(epath)

    for s, path_list in grouped_paths.items():
        if not path_list:
            continue

        # Build combined subgraph
        subG = nx.Graph()   # force undirected
        for epath in path_list:
            subG.add_edges_from(epath)

        n_nodes = subG.number_of_nodes()
        n_edges = subG.number_of_edges()

        # --- Auto-scaling rules ---
        base_node_size = 600
        base_font_size = 12
        base_edge_width = 1.2

        node_size = max(base_node_size, base_node_size * (10 / (n_nodes + 0)))
        font_size = max(base_font_size, base_font_size * (12 / (n_nodes + 0)))
        edge_width = max(base_edge_width, base_edge_width * (15 / (n_edges + 0)))
        arrowsize = max(base_arrowsize, base_arrowsize * (12 / (n_nodes + 0)))

        # Source node larger
        source_size = node_size * 1.5

        # Figure size proportional to graph size
        fig_w = max(5, min(12, 0.8 * n_nodes))
        fig_h = max(5, min(12, 0.8 * n_nodes))

        pos = nx.spring_layout(subG, seed=42, k=1.5 / (n_nodes ** 0.5))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.axis("off")

        # Draw normal nodes (exclude source)
        other_nodes = [n for n in subG.nodes() if n != s]
        nx.draw_networkx_nodes(
            subG, pos, ax=ax,
            nodelist=other_nodes,
            node_size=node_size,
            node_color="white", edgecolors="black"
        )

        # Draw source node (highlighted)
        nx.draw_networkx_nodes(
            subG, pos, ax=ax,
            nodelist=[s],
            node_size=source_size,
            node_color="red", edgecolors="black"
        )

        # Draw labels
        nx.draw_networkx_labels(subG, pos, ax=ax, font_size=font_size, font_color="black")

        # Draw edges
        drawn_edges = list(subG.edges())
        nx.draw_networkx_edges(
            subG,
            pos,
            ax=ax,
            edgelist=drawn_edges,
            width=[edge_width] * len(drawn_edges),
            arrows=is_directed,
            arrowstyle=arrowstyle if is_directed else "-",
            arrowsize=arrowsize if is_directed else 0,
            connectionstyle="arc3"
        )

        if plot_cap:

            # Edge labels (capacity from G)
            edge_labels = {}
            for (u, v) in drawn_edges:
                cap = None
                if G.has_edge(u, v):
                    cap = G[u][v].get("capacity") / scale
                if cap is None and (not is_directed) and G.has_edge(v, u):
                    cap = G[v][u].get("capacity") / scale
                if cap is None:
                    cap = "?"
                edge_labels[(u, v)] = str(cap)

            nx.draw_networkx_edge_labels(
                subG,
                pos,
                ax=ax,
                edge_labels=edge_labels,
                font_size=max(3, font_size - 2),
                label_pos=0.5,
                verticalalignment="bottom",
                clip_on=False
            )

        plt.tight_layout()
        fname = os.path.join(out_dir, f"source_{s}.png")
        plt.savefig(fname, bbox_inches="tight")
        plt.close(fig)

    print(f"All figures saved to: {out_dir}")


def draw_paths_in_group(
    G: nx.Graph,
    kpaths: dict,
    num_agents: int,
    out_dir: str = "ksp_plots",
    dpi: int = 240,
    arrowstyle: str = "-|>",
    base_arrowsize: int = 12,
    scale: float = 1.0,
    plot_cap: bool = False,
    plot_arrow: bool = True,
):
    """
    Draw candidate paths grouped by multiple source nodes per figure.
    Sources are split sequentially into num_agents groups.
    Each figure visualizes all paths originating from the sources in that group.
    """

    assert num_agents > 0
    os.makedirs(out_dir, exist_ok=True)
    is_directed = (G.is_directed() and plot_arrow)

    # -------- Step 1: group paths by source --------
    source_to_paths = defaultdict(list)
    for (s, t), path_list in kpaths.items():
        for epath in path_list:
            source_to_paths[s].append(epath)

    sources = sorted(source_to_paths.keys())
    n_sources = len(sources)
    assert num_agents <= n_sources, "num_agents must be <= number of sources"

    # -------- Step 2: sequentially split sources --------
    chunk_size = (n_sources + num_agents - 1) // num_agents
    source_groups = [
        sources[i:i + chunk_size]
        for i in range(0, n_sources, chunk_size)
    ]

    # -------- Step 3: draw per agent --------
    for agent_id, group_sources in enumerate(source_groups):
        path_list = []
        for s in group_sources:
            path_list.extend(source_to_paths[s])

        if not path_list:
            continue

        # Build combined subgraph
        subG = nx.Graph()
        for epath in path_list:
            subG.add_edges_from(epath)

        n_nodes = subG.number_of_nodes()
        n_edges = subG.number_of_edges()

        # -------- Auto scaling --------
        base_node_size = 600
        base_font_size = 12
        base_edge_width = 1.2

        node_size = max(base_node_size, base_node_size * (10 / (n_nodes + 1)))
        font_size = max(base_font_size, base_font_size * (12 / (n_nodes + 1)))
        edge_width = max(base_edge_width, base_edge_width * (15 / (n_edges + 1)))
        arrowsize = max(base_arrowsize, base_arrowsize * (12 / (n_nodes + 1)))

        source_size = node_size * 1.5

        fig_w = max(5, min(12, 0.8 * n_nodes))
        fig_h = max(5, min(12, 0.8 * n_nodes))

        pos = nx.spring_layout(subG, seed=42, k=1.5 / (n_nodes ** 0.5))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.axis("off")

        # -------- Draw nodes --------
        source_nodes = set(group_sources)
        other_nodes = [n for n in subG.nodes() if n not in source_nodes]

        nx.draw_networkx_nodes(
            subG, pos, ax=ax,
            nodelist=other_nodes,
            node_size=node_size,
            node_color="white",
            edgecolors="black"
        )

        nx.draw_networkx_nodes(
            subG, pos, ax=ax,
            nodelist=list(source_nodes),
            node_size=source_size,
            node_color="red",
            edgecolors="black"
        )

        nx.draw_networkx_labels(
            subG, pos, ax=ax,
            font_size=font_size,
            font_color="black"
        )

        # -------- Draw edges --------
        drawn_edges = list(subG.edges())
        nx.draw_networkx_edges(
            subG,
            pos,
            ax=ax,
            edgelist=drawn_edges,
            width=[edge_width] * len(drawn_edges),
            arrows=is_directed,
            arrowstyle=arrowstyle if is_directed else "-",
            arrowsize=arrowsize if is_directed else 0,
            connectionstyle="arc3"
        )

        # -------- Edge capacity labels --------
        if plot_cap:
            edge_labels = {}
            for (u, v) in drawn_edges:
                cap = None
                if G.has_edge(u, v):
                    cap = G[u][v].get("capacity")
                elif (not is_directed) and G.has_edge(v, u):
                    cap = G[v][u].get("capacity")
                edge_labels[(u, v)] = "?" if cap is None else str(cap / scale)

            nx.draw_networkx_edge_labels(
                subG,
                pos,
                ax=ax,
                edge_labels=edge_labels,
                font_size=max(3, font_size - 2),
                label_pos=0.5,
                verticalalignment="bottom",
                clip_on=False
            )

        plt.tight_layout()
        fname = os.path.join(out_dir, f"agent_{agent_id}.png")
        plt.savefig(fname, bbox_inches="tight")
        plt.close(fig)

    print(f"All figures saved to: {out_dir}")


def draw_paths_per_pair(
    G: nx.Graph,
    kpaths: dict,
    out_dir: str = "ksp_plots",
    dpi: int = 240,
    node_size: int = 600,
    font_size: int = 10,
    arrowstyle: str = "-|>",   # used if G is directed
    arrowsize: int = 12,
    edge_width: float = 1.2,
    scale: float = 1.0, 
    gap: float = 1.0,          # horizontal spacing between nodes in a path
    margin: float = 0.5,       # left/right margin for each row
    row_height: float = 0.75    # vertical space per row (smaller -> more compact)
):
    """
    Draws k-shortest paths for each source-destination pair in a networkx graph.

    Each path is drawn on its own row, aligned left-to-right.
    """

    os.makedirs(out_dir, exist_ok=True)
    is_directed = G.is_directed()

    def edges_to_nodes(edge_path):
        """Convert a list of edges [(u,v), (v,w), ...] into an ordered list of nodes [u, v, w, ...]."""
        if not edge_path:
            return []
        nodes = [edge_path[0][0], edge_path[0][1]]
        for (a, b) in edge_path[1:]:
            if nodes[-1] == a:
                nodes.append(b)
            elif nodes[-1] == b:  # tolerate reversed edge direction
                nodes.append(a)
            else:
                nodes += [a, b]
        return nodes

    for (s, t), path_list in kpaths.items():
        k = len(path_list)
        if k == 0:
            continue

        # Determine max path length (for figure width scaling)
        max_len = max(len(edges_to_nodes(epath)) for epath in path_list)

        # Figure size: width depends on max path length, height on number of paths * row_height
        fig_w = margin * 2 + max(2.0, (max_len - 1) * gap + 1.0)
        fig_h = k * row_height
        fig, axes = plt.subplots(k, 1, figsize=(fig_w, fig_h), dpi=dpi)
        if k == 1:
            axes = [axes]

        for idx, (ax, epath) in enumerate(zip(axes, path_list), start=1):
            ax.axis("off")

            # Build subgraph with just this path
            subG = nx.DiGraph() if is_directed else nx.Graph()
            subG.add_edges_from(epath)

            # Position nodes linearly along x-axis
            nodes = edges_to_nodes(epath)
            pos = {u: (margin + i * gap, 0.0) for i, u in enumerate(nodes)}

            # Draw nodes and labels
            nx.draw_networkx_nodes(subG, pos, ax=ax, node_size=node_size, node_color="white", edgecolors="black")
            nx.draw_networkx_labels(subG, pos, ax=ax, font_size=font_size)

            # Draw edges
            drawn_edges = list(subG.edges())
            nx.draw_networkx_edges(
                subG,
                pos,
                ax=ax,
                edgelist=drawn_edges,
                width=[edge_width] * len(drawn_edges),
                arrows=is_directed,
                arrowstyle=arrowstyle if is_directed else "-",
                arrowsize=arrowsize if is_directed else 0,
                connectionstyle="arc3"
            )

            # Draw edge labels (capacity from G)
            edge_labels = {}
            for (u, v) in drawn_edges:
                cap = None
                if G.has_edge(u, v):
                    cap = G[u][v].get("capacity") / scale
                if cap is None and (not is_directed) and G.has_edge(v, u):
                    cap = G[v][u].get("capacity") / scale
                if cap is None:
                    cap = "?"
                edge_labels[(u, v)] = str(cap)

            nx.draw_networkx_edge_labels(
                subG,
                pos,
                ax=ax,
                edge_labels=edge_labels,
                font_size=font_size,
                label_pos=0.5,
                verticalalignment="bottom",
                clip_on=False
            )

            # # Row title
            # ax.set_title(f"Path {idx}: {nodes}", fontsize=font_size + 1, pad=4)

        # Reduce vertical padding between rows
        plt.tight_layout(h_pad=0.3)
        fname = os.path.join(out_dir, f"{s}_{t}.png")
        plt.savefig(fname, bbox_inches="tight")
        plt.close(fig)

    print(f"All figures saved to: {out_dir}")


# =======================
if __name__ == "__main__":
    G = nx.DiGraph()
    G.add_edge(0,1,capacity=10)
    G.add_edge(0,2,capacity=8)
    G.add_edge(2,1,capacity=7)
    G.add_edge(0,3,capacity=9)
    G.add_edge(3,1,capacity=6)
    G.add_edge(0,4,capacity=5)
    G.add_edge(4,1,capacity=4)
    G.add_edge(1,2,capacity=3)
    G.add_edge(3,2,capacity=2)
    G.add_edge(4,2,capacity=1)
    G.add_edge(1,3,capacity=3)
    G.add_edge(2,3,capacity=2)
    G.add_edge(4,3,capacity=1)

    kpaths = {
        (0,1): [[(0,1)], [(0,2),(2,1)], [(0,3),(3,1)], [(0,4),(4,1)]],
        (0,2): [[(0,2)], [(0,1),(1,2)], [(0,3),(3,2)], [(0,4),(4,2)]],
        (0,3): [[(0,3)], [(0,1),(1,3)], [(0,2),(2,3)], [(0,4),(4,3)]],
    }

    draw_paths_per_pair(G, kpaths, out_dir="./ksp_demo_plots")
