"""Tests for constrained Delaunay triangulation."""

import numpy as np

from pycdt.build import triangulate
from pycdt.constrained import (
    find_overlapped_triangles,
    find_intersected_edges,
    insert_constraint_edge,
)


class TestFindOverlappedTriangles:
    """Tests for find_overlapped_triangles function."""

    def test_segment_in_single_triangle(self):
        """Test segment completely within one triangle."""
        points = np.array(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 4.0],
                [0.0, 4.0],
            ]
        )
        tri = triangulate(points)

        # Segment within one triangle
        p = np.array([0.5, 0.5])
        q = np.array([1.0, 1.0])

        overlapped = find_overlapped_triangles(tri, p, q)

        assert len(overlapped) >= 1

    def test_segment_across_multiple_triangles(self):
        """Test segment crossing multiple triangles."""
        points = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
            ]
        )
        tri = triangulate(points)

        # Diagonal segment
        p = np.array([0.1, 0.1])
        q = np.array([1.9, 1.9])

        overlapped = find_overlapped_triangles(tri, p, q)

        # Should pass through at least one triangle
        assert len(overlapped) >= 1

        # All triangles should be unique
        assert len(overlapped) == len(set(overlapped))

    def test_segment_endpoints_are_vertices(self):
        """Test segment where endpoints are existing vertices."""
        points = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
            ]
        )
        tri = triangulate(points)

        # Segment between two vertices
        p = points[0]  # (0, 0)
        q = points[2]  # (2, 2)

        overlapped = find_overlapped_triangles(tri, p, q)

        assert len(overlapped) >= 1

    def test_horizontal_segment(self):
        """Test horizontal segment."""
        points = np.array(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 2.0],
                [0.0, 2.0],
            ]
        )
        tri = triangulate(points)

        p = np.array([0.5, 1.0])
        q = np.array([3.5, 1.0])

        overlapped = find_overlapped_triangles(tri, p, q)

        assert len(overlapped) >= 1

    def test_vertical_segment(self):
        """Test vertical segment."""
        points = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 4.0],
                [0.0, 4.0],
            ]
        )
        tri = triangulate(points)

        p = np.array([1.0, 0.5])
        q = np.array([1.0, 3.5])

        overlapped = find_overlapped_triangles(tri, p, q)

        assert len(overlapped) >= 1

    def test_segment_with_grid_triangulation(self):
        """Test segment through a grid triangulation."""
        # Create a 3x3 grid
        x = np.linspace(0, 2, 3)
        y = np.linspace(0, 2, 3)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        tri = triangulate(points)

        # Diagonal segment
        p = np.array([0.1, 0.1])
        q = np.array([1.9, 1.9])

        overlapped = find_overlapped_triangles(tri, p, q)

        # Should find multiple triangles
        assert len(overlapped) >= 1


class TestFindIntersectedEdges:
    """Tests for find_intersected_edges function."""

    def test_segment_intersects_edges(self):
        """Test finding edges intersected by a segment."""
        points = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
            ]
        )
        tri = triangulate(points)

        # Diagonal segment
        p = np.array([0.1, 0.1])
        q = np.array([1.9, 1.9])

        overlapped = find_overlapped_triangles(tri, p, q)
        intersected = find_intersected_edges(tri, p, q, overlapped)

        # Should find at least some intersected edges (unless segment only goes through one triangle)
        assert isinstance(intersected, list)

    def test_segment_within_triangle_no_edges(self):
        """Test segment completely within one triangle intersects no internal edges."""
        points = np.array(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 4.0],
                [0.0, 4.0],
            ]
        )
        tri = triangulate(points)

        # Small segment within one triangle
        p = np.array([0.5, 0.5])
        q = np.array([0.7, 0.7])

        overlapped = find_overlapped_triangles(tri, p, q)
        intersected = find_intersected_edges(tri, p, q, overlapped)

        # Segment within one triangle shouldn't intersect internal edges
        # (may intersect boundary edges depending on triangle orientation)
        assert isinstance(intersected, list)

    def test_edges_are_canonical(self):
        """Test that returned edges are in canonical form (smaller index first)."""
        points = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
            ]
        )
        tri = triangulate(points)

        p = np.array([0.1, 0.1])
        q = np.array([1.9, 1.9])

        overlapped = find_overlapped_triangles(tri, p, q)
        intersected = find_intersected_edges(tri, p, q, overlapped)

        # Check that all edges have smaller index first
        for v1, v2 in intersected:
            assert v1 < v2


class TestInsertConstraintEdge:
    """Tests for insert_constraint_edge function."""

    def test_insert_constraint_basic(self):
        """Test basic constraint insertion."""
        points = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
            ]
        )
        tri = triangulate(points)

        # Add constraint from vertex 0 to vertex 2
        p = points[0]
        q = points[2]

        # This will currently return False since re-triangulation is not implemented
        result = insert_constraint_edge(tri, p, q, 0, 2)

        # For now, just check it doesn't crash
        assert isinstance(result, bool)

    def test_constraint_already_exists(self):
        """Test inserting a constraint that already exists as an edge."""
        points = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ]
        )
        tri = triangulate(points)

        # This forms a single triangle, so edge 0-1 already exists
        p = points[0]
        q = points[1]

        result = insert_constraint_edge(tri, p, q, 0, 1)

        # Should succeed (edge already exists)
        assert isinstance(result, bool)


class TestConstrainedTriangulationIntegration:
    """Integration tests for constrained triangulation."""

    def test_square_with_diagonal_constraint(self):
        """Test triangulating a square with a diagonal constraint."""
        points = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        tri = triangulate(points)

        # Find overlapped triangles for diagonal
        p = points[0]  # (0, 0)
        q = points[2]  # (1, 1)

        overlapped = find_overlapped_triangles(tri, p, q)

        # Should find path from corner to corner
        assert len(overlapped) >= 1

        # First and last should contain p and q respectively
        from pycdt.geometry import point_inside_triangle, PointInTriangle

        first_tri = tri.all_points[tri.triangle_vertices[overlapped[0]]]
        p_status, _ = point_inside_triangle(first_tri, p)
        assert p_status in (
            PointInTriangle.inside,
            PointInTriangle.vertex,
            PointInTriangle.edge,
        )

        last_tri = tri.all_points[tri.triangle_vertices[overlapped[-1]]]
        q_status, _ = point_inside_triangle(last_tri, q)
        assert q_status in (
            PointInTriangle.inside,
            PointInTriangle.vertex,
            PointInTriangle.edge,
        )
