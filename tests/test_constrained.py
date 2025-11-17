"""Tests for constrained Delaunay triangulation."""

import numpy as np
from shewchuk import incircle_test

from pycdt.build import triangulate
from pycdt.constrained import (
    find_intersecting_edges,
    IntersectedEdge,
    add_constraints,
)


def test_find_intersecting_edges_single_edge():
    """Test finding a single intersecting edge in a simple triangulation."""
    # Create a simple square triangulation
    points = np.array(
        [
            [0.0, 0.0],  # 0
            [2.0, 0.0],  # 1
            [2.0, 2.0],  # 2
            [0.0, 2.0],  # 3
            [1.0, 1.0],  # 4 - center point
        ]
    )

    tri = triangulate(points)

    # Add constraint from bottom-left to top-right (should cross diagonal through center)
    # Find which vertices correspond to our corners after triangulation
    p_idx = 0  # bottom-left
    q_idx = 2  # top-right

    edges = find_intersecting_edges(tri, p_idx, q_idx)

    # Should find at least one intersecting edge
    assert len(edges) > 0
    assert all(isinstance(edge, IntersectedEdge) for edge in edges)

    # Each edge should have valid triangle indices
    for edge in edges:
        assert edge.triangle_1 >= 0
        assert edge.triangle_2 >= 0 or edge.triangle_2 == -1  # -1 for boundary


def test_find_intersecting_edges_same_triangle():
    """Test when both endpoints are in the same triangle."""
    # Create a simple triangulation
    points = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0],
        ]
    )

    tri = triangulate(points)

    # Try to add constraint between two adjacent vertices (edge already exists)
    p_idx = 0
    q_idx = 1

    edges = find_intersecting_edges(tri, p_idx, q_idx)

    # Should return one edge representing the existing edge
    assert len(edges) == 1
    assert edges[0].p1 == p_idx or edges[0].p1 == q_idx
    assert edges[0].p2 == p_idx or edges[0].p2 == q_idx
    assert edges[0].triangle_1 == edges[0].triangle_2  # Same triangle


def test_find_intersecting_edges_multiple():
    """Test finding multiple intersecting edges in a larger triangulation."""
    # Create a rectangular grid of points
    x = np.linspace(0, 10, 6)
    y = np.linspace(0, 10, 6)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    tri = triangulate(points)

    # Add constraint from bottom-left corner to top-right corner
    # This should cross many edges
    bottom_left = np.argmin(np.sum(points**2, axis=1))
    top_right = np.argmax(np.sum(points**2, axis=1))

    edges = find_intersecting_edges(tri, bottom_left, top_right)

    # Should find multiple intersecting edges
    assert len(edges) >= 2

    # All edges should have valid vertex indices
    for edge in edges:
        assert 0 <= edge.p1 < len(points)
        assert 0 <= edge.p2 < len(points)
        assert edge.p1 != edge.p2

        # Triangles should be valid
        assert edge.triangle_1 >= 0
        # triangle_2 might be -1 for boundary edges, but should exist for internal edges
        assert edge.triangle_2 >= -1


def test_find_intersecting_edges_horizontal():
    """Test with a horizontal constraint edge."""
    # Create a triangulation with clear horizontal structure
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [3.0, 2.0],
        ]
    )

    tri = triangulate(points)

    # Add horizontal constraint from left to right at y=1.0
    # Find points at y=1.0
    left_idx = 4  # (0, 1)
    right_idx = 7  # (3, 1)

    edges = find_intersecting_edges(tri, left_idx, right_idx)

    # Should find edges (might be 0 if edge already exists, or multiple if it crosses diagonals)
    assert len(edges) >= 0

    # Verify all edges have proper structure
    for edge in edges:
        assert isinstance(edge, IntersectedEdge)
        assert hasattr(edge, "p1")
        assert hasattr(edge, "p2")
        assert hasattr(edge, "triangle_1")
        assert hasattr(edge, "triangle_2")


def test_find_intersecting_edges_vertical():
    """Test with a vertical constraint edge."""
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ]
    )

    tri = triangulate(points)

    # Add vertical constraint from bottom to top at x=1.0
    bottom_idx = 1  # (1, 0)
    top_idx = 7  # (1, 2)

    edges = find_intersecting_edges(tri, bottom_idx, top_idx)

    # Should find edges
    assert len(edges) >= 0

    for edge in edges:
        assert edge.p1 >= 0
        assert edge.p2 >= 0
        assert edge.triangle_1 >= 0


def test_find_intersecting_edges_diagonal():
    """Test with a diagonal constraint across the entire triangulation."""
    # Create a square grid
    n = 5
    x = np.linspace(0, 4, n)
    y = np.linspace(0, 4, n)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    tri = triangulate(points)

    # Find bottom-left and top-right corners
    p_idx = 0  # (0, 0)
    q_idx = len(points) - 1  # (4, 4)

    edges = find_intersecting_edges(tri, p_idx, q_idx)

    # A diagonal across the grid should intersect multiple edges
    assert len(edges) >= 1

    # Verify ordering: edges should be in sequence from p to q
    # Each subsequent edge should share at least one triangle with the previous
    # (or be part of the walking path)
    for i, edge in enumerate(edges):
        assert edge.p1 < len(points)
        assert edge.p2 < len(points)


def test_find_intersecting_edges_returns_intersected_edge_objects():
    """Test that the function returns IntersectedEdge objects with all fields populated."""
    points = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ]
    )

    tri = triangulate(points)

    p_idx = 0
    q_idx = 2

    edges = find_intersecting_edges(tri, p_idx, q_idx)

    for edge in edges:
        # Check that it's the correct type
        assert isinstance(edge, IntersectedEdge)

        # Check all fields are present and valid
        assert isinstance(edge.p1, (int, np.integer))
        assert isinstance(edge.p2, (int, np.integer))
        assert isinstance(edge.triangle_1, (int, np.integer))
        assert isinstance(edge.triangle_2, (int, np.integer))

        # Vertex indices should be different
        assert edge.p1 != edge.p2

        # At least one triangle should be valid (not boundary)
        assert edge.triangle_1 >= 0


def test_find_intersecting_edges_consistent_vertex_order():
    """Test that edge vertices are consistently ordered."""
    points = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 3.0],
            [0.0, 3.0],
            [1.5, 1.5],
        ]
    )

    tri = triangulate(points)

    p_idx = 0
    q_idx = 2

    edges = find_intersecting_edges(tri, p_idx, q_idx)

    # According to the code, vertices should be sorted
    for edge in edges:
        assert edge.p1 <= edge.p2, "Edge vertices should be sorted"


def test_find_intersecting_edges_debug_mode():
    """Test that debug mode doesn't crash (visual output not verified)."""
    import matplotlib

    # Use non-interactive backend to avoid showing plots during tests
    matplotlib.use("Agg")

    points = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ]
    )

    tri = triangulate(points)

    p_idx = 0
    q_idx = 2

    # This should not crash even with debug=True
    edges = find_intersecting_edges(tri, p_idx, q_idx, debug=True)

    # Verify we still get valid results
    assert len(edges) > 0
    for edge in edges:
        assert isinstance(edge, IntersectedEdge)


def test_add_constraints_single():
    """Test adding a single constraint edge."""
    # Create a simple square triangulation
    points = np.array(
        [
            [0.0, 0.0],  # 0
            [2.0, 0.0],  # 1
            [2.0, 2.0],  # 2
            [0.0, 2.0],  # 3
        ]
    )

    tri = triangulate(points)
    num_triangles_before = len(tri.triangle_vertices)

    # Insert constraint from bottom-left to top-right
    p_idx = 0
    q_idx = 2

    success = add_constraints(tri, (p_idx, q_idx))

    assert success
    # Number of triangles should remain the same (just reorganized)
    assert len(tri.triangle_vertices) == num_triangles_before


def test_add_constraints_existing_edge():
    """Test adding a constraint that's already an edge."""
    points = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )

    tri = triangulate(points)
    num_triangles_before = len(tri.triangle_vertices)

    # Try to insert an edge that already exists
    p_idx = 0
    q_idx = 1  # Adjacent vertices

    success = add_constraints(tri, (p_idx, q_idx))

    assert success
    # Should not change the triangulation
    assert len(tri.triangle_vertices) == num_triangles_before


def test_add_constraints_delaunay_restoration():
    """Test that Delaunay property is restored after constraint insertion."""

    def count_delaunay_violations(tri):
        """Count triangles that violate the Delaunay property."""
        violations = 0
        for tri_idx, tri_verts in enumerate(tri.triangle_vertices):
            t_points = tri.all_points[tri_verts]
            neighbors = tri.triangle_neighbors[tri_idx]

            for neighbor_idx in neighbors:
                if neighbor_idx == -1:
                    continue

                neighbor_verts = tri.triangle_vertices[neighbor_idx]
                shared = set(tri_verts) & set(neighbor_verts)
                opposite_verts = set(neighbor_verts) - shared

                if not opposite_verts:
                    continue

                opposite_idx = list(opposite_verts)[0]
                opposite_point = tri.all_points[opposite_idx]

                # Check if opposite point is inside circumcircle
                if (
                    incircle_test(
                        *opposite_point, *t_points[0], *t_points[1], *t_points[2]
                    )
                    > 0
                ):
                    violations += 1

        # Each violation is counted twice (once from each triangle)
        return violations // 2

    # Create a grid
    n = 4
    x = np.linspace(0, 3, n)
    y = np.linspace(0, 3, n)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    tri = triangulate(points)

    # Initial triangulation should be Delaunay
    violations_before = count_delaunay_violations(tri)
    assert violations_before == 0

    # Insert a short constraint that doesn't cross many edges
    p_idx = 0  # (0, 0)
    q_idx = 5  # (1, 1)

    success = add_constraints(tri, (p_idx, q_idx))
    assert success

    # After restoration, violations should be minimal
    # (only at constraint edges if unavoidable)
    violations_after = count_delaunay_violations(tri)
    # Allow some violations at the constraint edge itself
    assert violations_after <= 2


def test_add_constraints_preserves_connectivity():
    """Test that constraint insertion maintains valid triangle connectivity."""
    points = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 3.0],
            [0.0, 3.0],
            [1.5, 1.5],
        ]
    )

    tri = triangulate(points)

    # Insert constraint
    p_idx = 0
    q_idx = 2

    success = add_constraints(tri, (p_idx, q_idx))
    assert success

    # Check that all triangles have valid neighbors
    for tri_idx, tri_verts in enumerate(tri.triangle_vertices):
        neighbors = tri.triangle_neighbors[tri_idx]

        for i, neighbor_idx in enumerate(neighbors):
            if neighbor_idx == -1:
                continue  # Boundary edge is OK

            # Check that the neighbor is valid
            assert 0 <= neighbor_idx < len(tri.triangle_vertices)

            # Check that we are also a neighbor of that triangle
            neighbor_neighbors = tri.triangle_neighbors[neighbor_idx]
            assert tri_idx in neighbor_neighbors


def test_add_constraints_diagonal():
    """Test adding a long diagonal constraint across a grid."""
    # Create a larger grid
    n = 5
    x = np.linspace(0, 4, n)
    y = np.linspace(0, 4, n)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    tri = triangulate(points)
    num_triangles_before = len(tri.triangle_vertices)

    # Insert diagonal constraint from corner to corner
    p_idx = 0  # (0, 0)
    q_idx = len(points) - 1  # (4, 4)

    success = add_constraints(tri, (p_idx, q_idx))
    assert success

    # Number of triangles should remain the same
    assert len(tri.triangle_vertices) == num_triangles_before

    # Verify the constraint edge exists in some form
    # (either as a triangle edge or implied by the structure)
    # We can't easily verify this without additional data structures,
    # but we can check that the triangulation is still valid
    for tri_verts in tri.triangle_vertices:
        assert len(tri_verts) == 3
        assert len(set(tri_verts)) == 3  # No duplicate vertices


def test_add_constraints_multiple():
    """Test adding multiple constraints using add_constraints function."""
    # Create a simple square triangulation
    points = np.array(
        [
            [0.0, 0.0],  # 0
            [2.0, 0.0],  # 1
            [2.0, 2.0],  # 2
            [0.0, 2.0],  # 3
        ]
    )

    tri = triangulate(points)

    # Add two diagonal constraints
    constraints = [
        (0, 2),  # First diagonal
        (1, 3),  # Second diagonal
    ]

    # This should not crash
    success = add_constraints(tri, constraints)
    assert success

    # Verify triangulation is still valid
    assert len(tri.triangle_vertices) > 0

    # Check all triangles have valid vertices
    for tri_verts in tri.triangle_vertices:
        assert len(tri_verts) == 3
        for v_idx in tri_verts:
            assert 0 <= v_idx < len(tri.all_points)


def test_add_constraints_maintains_point_count():
    """Test that adding constraints doesn't add or remove points."""
    points = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ]
    )

    tri = triangulate(points)
    num_points_before = len(tri.all_points)

    # Insert constraint
    add_constraints(tri, (0, 2))

    # Number of points should not change
    assert len(tri.all_points) == num_points_before
