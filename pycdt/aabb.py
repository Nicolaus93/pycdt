from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, TypeAlias, Self

import numpy as np
from numpy.typing import NDArray
from loguru import logger

from pycdt.delaunay import Triangulation
from pycdt.geometry import PointInTriangle

Bbox: TypeAlias = tuple[float, float, float, float]


@dataclass
class AABBNode:
    bbox: Bbox
    left: Optional["AABBNode"] = None
    right: Optional["AABBNode"] = None
    triangles: Optional[list[int]] = None
    parent: Optional["AABBNode"] = None

    def is_leaf(self):
        return self.triangles is not None


def merge_bbox(a: Bbox, b: Bbox) -> Bbox:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def bbox_area(b: Bbox) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


def enlargement_cost(parent_bbox: Bbox, tri_bbox: Bbox):
    merged = merge_bbox(parent_bbox, tri_bbox)
    enlargement = bbox_area(merged) - bbox_area(parent_bbox)
    if enlargement < 0:
        raise ValueError(f"Enlargement cost {enlargement} is less than zero")
    return enlargement


class AABBTree:
    def __init__(self, max_leaf_size: int = 20) -> None:
        self.root: Optional[AABBNode] = None
        self.tri_bboxes: dict[int, Bbox] = {}
        self.max_leaf_size = max_leaf_size

    def _compute_group_bbox(self, indices: list[int]) -> Bbox:
        xmins, ymins, xmaxs, ymaxs = zip(*(self.tri_bboxes[i] for i in indices))
        return min(xmins), min(ymins), max(xmaxs), max(ymaxs)

    def _split_leaf(self, node: AABBNode) -> None:
        tris = node.triangles or []
        if not tris:
            return
        node.triangles = None

        # Compute axis & split midpoint
        xmins, ymins, xmaxs, ymaxs = zip(*(self.tri_bboxes[i] for i in tris))
        bbox = (min(xmins), min(ymins), max(xmaxs), max(ymaxs))
        dx = bbox[2] - bbox[0]
        dy = bbox[3] - bbox[1]
        axis = 0 if dx > dy else 1

        centers = [
            (self.tri_bboxes[i][axis] + self.tri_bboxes[i][axis + 2]) / 2 for i in tris
        ]
        mid = sorted(centers)[len(centers) // 2]

        left_indices = [
            i
            for i in tris
            if (self.tri_bboxes[i][axis] + self.tri_bboxes[i][axis + 2]) / 2 <= mid
        ]
        right_indices = [i for i in tris if i not in left_indices]

        # Fallback: degenerate partition → index-based spatial split
        if not left_indices or not right_indices:
            tris_sorted = sorted(tris, key=lambda i: centers[tris.index(i)])
            half = len(tris_sorted) // 2
            left_indices, right_indices = tris_sorted[:half], tris_sorted[half:]

        left_bbox = self._compute_group_bbox(left_indices)
        right_bbox = self._compute_group_bbox(right_indices)

        node.left = AABBNode(left_bbox, triangles=left_indices, parent=node)
        node.right = AABBNode(right_bbox, triangles=right_indices, parent=node)

    def insert(self, tri_index: int, tri_bbox: Bbox) -> None:
        """Insert a triangle bounding box into the tree."""
        self.tri_bboxes[tri_index] = tri_bbox

        if self.root is None:
            self.root = AABBNode(bbox=tri_bbox, triangles=[tri_index])
            return

        node = self.root
        # Descend to the leaf with minimum enlargement
        while not node.is_leaf():
            cost_left = (
                enlargement_cost(node.left.bbox, tri_bbox)
                if node.left
                else float("inf")
            )
            cost_right = (
                enlargement_cost(node.right.bbox, tri_bbox)
                if node.right
                else float("inf")
            )
            node = node.left if cost_left < cost_right else node.right

        # Insert into leaf
        node.triangles.append(tri_index)
        node.bbox = merge_bbox(node.bbox, tri_bbox)  # type: ignore[reportOptionalMemberAccess]

        # Split if necessary
        if len(node.triangles) > self.max_leaf_size:
            logger.debug("Splitting aabb tree")
            self._split_leaf(node)

        # Update ancestors
        n = node.parent  # type: ignore[reportOptionalMemberAccess]
        while n:
            n.bbox = merge_bbox(n.left.bbox, n.right.bbox)  # type: ignore[reportOptionalMemberAccess]
            n = n.parent

    def find(
        self,
        pt: NDArray[np.floating],
        triangles: dict[int, NDArray[np.floating]],
        point_inside_triangle: Callable[
            [NDArray[np.floating], NDArray[np.floating]],
            tuple[PointInTriangle, Optional[int]],
        ],
    ) -> Optional[int]:
        """
        Find the index of the triangle that contains a given 2D point.

        Parameters
        ----------
        pt : NDArray[np.floating]
            The query point as a (2,) array.
        triangles : dict[int, NDArray[np.floating]]
            Mapping from triangle index to vertex array of shape (3, 2).
        point_inside_triangle : callable
            A function returning (PointInTriangle, Optional[int]) for a triangle and a point.
        include_edges : bool, optional
            If False, points lying exactly on edges are ignored (treated as outside).

        Returns
        -------
        Optional[int]
            The index of the containing triangle, or None if no triangle contains the point.
        """

        def _find(node: Optional[AABBNode]) -> Optional[int]:
            if node is None:
                return None

            x, y = pt
            xmin, ymin, xmax, ymax = node.bbox
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                return None

            if node.is_leaf():
                for i in node.triangles:  # type: ignore[reportOptionalIterable]
                    status, _ = point_inside_triangle(triangles[i], pt)
                    if status in (
                        PointInTriangle.inside,
                        PointInTriangle.edge,
                        PointInTriangle.vertex,
                    ):
                        return i
                return None

            return _find(node.left) or _find(node.right)

        return _find(self.root)

    @classmethod
    def from_triangulation(cls, triangulation: Triangulation) -> Self:
        """
        Build an AABBTree from a Triangulation instance.

        Each triangle's bounding box is inserted into the tree.

        Parameters
        ----------
        triangulation : Triangulation
            The triangulation containing all points and triangles.

        Returns
        -------
        AABBTree
            A fully built AABB tree for the current triangulation.
        """
        tree = cls()
        num_tris = len(triangulation.triangle_vertices)
        if num_tris == 0:
            return tree

        for i, v_indices in enumerate(triangulation.triangle_vertices):
            tri = triangulation.all_points[v_indices]
            xs, ys = tri[:, 0], tri[:, 1]
            bbox = (xs.min(), ys.min(), xs.max(), ys.max())
            tree.insert(i, bbox)
        return tree
