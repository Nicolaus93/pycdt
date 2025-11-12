import numpy as np
import typing

from pycdt.delaunay import Triangulation

if typing.TYPE_CHECKING:
    from pycdt.constrained import IntersectedEdge


def _plot_intersecting_edges(
    triangulation: Triangulation,
    p_idx: int,
    q_idx: int,
    intersecting: list["IntersectedEdge"],
) -> None:
    """
    Visualize the constraint segment and intersected edges/triangles.

    Parameters
    ----------
    triangulation : Triangulation
        The triangulation
    p_idx : int
        Start vertex index of constraint segment
    q_idx : int
        End vertex index of constraint segment
    intersecting : list[IntersectedEdge]
        List of intersected edges
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(12, 10))

    # Get all points and triangles
    points = triangulation.all_points
    triangles = triangulation.triangle_vertices

    # Collect all triangles that are intersected
    intersected_triangles = set()
    for edge in intersecting:
        if edge.triangle_1 >= 0:
            intersected_triangles.add(edge.triangle_1)
        if edge.triangle_2 >= 0:
            intersected_triangles.add(edge.triangle_2)

    # Plot all triangles with clear edges and triangle numbers
    for tri_idx, tri_verts in enumerate(triangles):
        tri_points = points[tri_verts]
        if tri_idx in intersected_triangles:
            # Highlight intersected triangles in orange
            poly = Polygon(
                tri_points,
                alpha=0.5,
                facecolor="orange",
                edgecolor="black",
                linewidth=1.5,
                zorder=2,
            )
        else:
            # Normal triangles - white with black edges
            poly = Polygon(
                tri_points,
                alpha=1.0,
                facecolor="white",
                edgecolor="darkgray",
                linewidth=0.8,
                zorder=1,
            )
        ax.add_patch(poly)

        # Add triangle number at centroid
        centroid = np.mean(tri_points, axis=0)
        text_color = "darkred" if tri_idx in intersected_triangles else "darkblue"
        ax.text(
            centroid[0],
            centroid[1],
            str(tri_idx),
            fontsize=8,
            ha="center",
            va="center",
            color=text_color,
            fontweight="bold" if tri_idx in intersected_triangles else "normal",
            zorder=3,
            bbox=dict(
                boxstyle="circle,pad=0.1",
                facecolor="white",
                edgecolor=text_color,
                alpha=0.8,
                linewidth=0.5,
            ),
        )

    # Plot all vertices of the triangulation with vertex numbers
    for v_idx, point in enumerate(points):
        # Skip start and end points as they have special markers
        if v_idx == p_idx or v_idx == q_idx:
            continue

        # Plot vertex as small dot
        ax.plot(point[0], point[1], "k.", markersize=3, zorder=3, alpha=0.5)

        # Add vertex number label
        ax.text(
            point[0],
            point[1],
            f" {v_idx}",
            fontsize=7,
            ha="left",
            va="bottom",
            color="darkgreen",
            fontweight="bold",
            zorder=7,
            alpha=0.8,
        )

    # Plot intersected edges in red with thicker lines
    for i, edge in enumerate(intersecting):
        v1_coords = points[edge.p1]
        v2_coords = points[edge.p2]
        ax.plot(
            [v1_coords[0], v2_coords[0]],
            [v1_coords[1], v2_coords[1]],
            "r-",
            linewidth=3,
            alpha=0.8,
            label="Intersected edges" if i == 0 else "",
            zorder=4,
        )

    # Plot the constraint segment in blue with thick line
    p = points[p_idx]
    q = points[q_idx]
    ax.plot(
        [p[0], q[0]],
        [p[1], q[1]],
        color="blue",
        linewidth=4,
        marker="o",
        markersize=10,
        markerfacecolor="blue",
        markeredgecolor="white",
        markeredgewidth=2,
        label="Constraint segment",
        zorder=5,
        alpha=0.9,
    )

    # Plot start and end points with labels
    ax.plot(
        p[0],
        p[1],
        "o",
        color="green",
        markersize=12,
        markeredgecolor="white",
        markeredgewidth=2,
        label=f"Start (v{p_idx})",
        zorder=6,
    )
    ax.plot(
        q[0],
        q[1],
        "o",
        color="purple",
        markersize=12,
        markeredgecolor="white",
        markeredgewidth=2,
        label=f"End (v{q_idx})",
        zorder=6,
    )

    # Add text labels for start and end
    ax.text(
        p[0],
        p[1],
        f"  p({p_idx})",
        fontsize=10,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.text(
        q[0],
        q[1],
        f"  q({q_idx})",
        fontsize=10,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Set axis properties
    ax.set_aspect("equal")
    ax.legend(loc="best", framealpha=0.9)
    ax.set_title(
        f"Constrained Delaunay Triangulation\n"
        f"Constraint edge: v{p_idx} → v{q_idx} | "
        f"Intersected edges: {len(intersecting)} | "
        f"Intersected triangles: {len(intersected_triangles)}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.2, linestyle="--")

    plt.tight_layout()
    plt.show()
