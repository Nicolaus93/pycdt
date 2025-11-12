"""Unit tests for triangulation query functions (pycdt/query.py and pycdt/build.py)."""

import numpy as np
import pytest

from pycdt.build import triangulate, find_containing_triangle
from pycdt.constrained import segments_intersect


class TestSegmentsIntersect:
    """Tests for segments_intersect function."""

    def test_parallel_segments_no_intersection(self):
        """Test parallel segments that don't intersect."""
        p1 = (0.0, 0.0)
        p2 = (1.0, 0.0)
        q1 = (0.0, 1.0)
        q2 = (1.0, 1.0)
        assert not segments_intersect(p1, p2, q1, q2)

    def test_crossing_segments_intersect(self):
        """Test segments that cross each other."""
        p1 = (0.0, 0.0)
        p2 = (1.0, 1.0)
        q1 = (0.0, 1.0)
        q2 = (1.0, 0.0)
        assert segments_intersect(p1, p2, q1, q2)

    def test_touching_endpoints(self):
        """Test segments that share an endpoint."""
        p1 = (0.0, 0.0)
        p2 = (1.0, 1.0)
        q1 = (1.0, 1.0)
        q2 = (2.0, 0.0)
        assert segments_intersect(p1, p2, q1, q2)

    def test_collinear_overlapping_segments(self):
        """Test collinear segments that overlap."""
        p1 = (0.0, 0.0)
        p2 = (2.0, 0.0)
        q1 = (1.0, 0.0)
        q2 = (3.0, 0.0)
        assert segments_intersect(p1, p2, q1, q2)

    def test_collinear_non_overlapping_segments(self):
        """Test collinear segments that don't overlap."""
        p1 = (0.0, 0.0)
        p2 = (1.0, 0.0)
        q1 = (2.0, 0.0)
        q2 = (3.0, 0.0)
        assert not segments_intersect(p1, p2, q1, q2)

    def test_t_intersection(self):
        """Test T-shaped intersection where one segment ends on another."""
        p1 = (0.0, 0.0)
        p2 = (2.0, 0.0)
        q1 = (1.0, 0.0)
        q2 = (1.0, 1.0)
        assert segments_intersect(p1, p2, q1, q2)

    def test_perpendicular_non_intersecting(self):
        """Test perpendicular segments that don't intersect."""
        p1 = (0.0, 0.0)
        p2 = (1.0, 0.0)
        q1 = (2.0, -1.0)
        q2 = (2.0, 1.0)
        assert not segments_intersect(p1, p2, q1, q2)

    def test_diagonal_segments(self):
        """Test diagonal segments that intersect in the middle."""
        p1 = (-1.0, -1.0)
        p2 = (1.0, 1.0)
        q1 = (-1.0, 1.0)
        q2 = (1.0, -1.0)
        assert segments_intersect(p1, p2, q1, q2)


class TestFindContainingTriangle:
    """Tests for find_containing_triangle function from build.py."""

    def test_find_point_in_center(self):
        """Test finding a triangle containing a point in the center."""
        points = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        tri = triangulate(points)

        # Point in center of square
        point = np.array([0.5, 0.5])
        result = find_containing_triangle(tri, point, 0)

        assert result.idx >= 0
        # Verify the point is actually in the found triangle
        tri_verts = tri.all_points[tri.triangle_vertices[result.idx]]
        from pycdt.geometry import point_inside_triangle, PointInTriangle

        status, _ = point_inside_triangle(tri_verts, point)
        assert status in (
            PointInTriangle.inside,
            PointInTriangle.edge,
            PointInTriangle.vertex,
        )

    def test_find_point_on_vertex(self):
        """Test finding a triangle containing a point on a vertex."""
        points = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        tri = triangulate(points)

        # Point on vertex
        point = np.array([0.0, 0.0])
        result = find_containing_triangle(tri, point, 0)

        assert result.idx >= 0

    def test_find_point_outside_raises_error(self):
        """Test that finding a point outside raises an error."""
        points = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        tri = triangulate(points)

        # Point far outside - may raise ValueError or IndexError depending on
        # whether neighbor relationships are broken in finalized triangulation
        point = np.array([10.0, 10.0])
        with pytest.raises((ValueError, IndexError)):
            find_containing_triangle(tri, point, 0)

    def test_find_point_near_edge(self):
        """Test finding a triangle containing a point near an edge."""
        points = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
            ]
        )
        tri = triangulate(points)

        # Point near an edge
        point = np.array([1.0, 0.1])
        result = find_containing_triangle(tri, point, 0)

        assert result.idx >= 0
