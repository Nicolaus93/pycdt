"""Example: Insert a long diagonal constraint across a grid.

This example demonstrates inserting a constraint edge that spans the entire
diagonal of a regular grid of points.
"""

import numpy as np

from pycdt.build import triangulate
from pycdt.constrained import add_constraints


def main():
    """Example: Long diagonal constraint across a grid."""
    print("\n" + "=" * 70)
    print("GRID DIAGONAL CONSTRAINT EXAMPLE")
    print("=" * 70 + "\n")

    # Create a 6x6 grid of points
    n = 6
    x = np.linspace(0, 5, n)
    y = np.linspace(0, 5, n)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    print(f"Number of points: {len(points)} ({n}×{n} grid)")

    # Create initial triangulation
    tri = triangulate(points)
    print(f"Number of triangles: {len(tri.triangle_vertices)}")

    # Plot before constraint
    print("\nPlotting triangulation before constraint...")
    tri.plot(
        show=False,
        title="Grid Before Constraint",
        point_labels=True,
        fontsize=8,
    )

    # Insert diagonal constraint from corner to corner
    p_idx = 30  # top-left corner
    q_idx = 5  # bottom right corner

    print(f"\nAdding constraint edge {p_idx} -> {q_idx}")
    print(f"  From point: {points[p_idx]} (bottom-left corner)")
    print(f"  To point: {points[q_idx]} (top-right corner)")

    success = add_constraints(tri, [(p_idx, q_idx)])

    if success:
        print("  ✓ Constraint successfully added")
    else:
        print("  ✗ Failed to add constraint")

    # Plot after constraint
    print("\nPlotting triangulation after constraint...")
    tri.plot(
        show=True,
        title="Grid After Constraint",
        point_labels=True,
        fontsize=8,
        constraints=[(p_idx, q_idx)],
    )

    print(f"\nAfter constraint insertion: {len(tri.triangle_vertices)} triangles")
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
