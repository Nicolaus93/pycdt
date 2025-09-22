import numpy as np
from loguru import logger
from numpy._typing import NDArray

from pycdt.delaunay import Triangulation
from pycdt.topology import lawson_swapping, reorder_neighbors_for_triangle
from pycdt.geometry import point_inside_triangle, ensure_ccw_triangle, is_inside_domain


def find_containing_triangle(
    triangulation: Triangulation,
    point: NDArray[np.floating],
    last_triangle_idx: int,
) -> int:
    """
    Implementation of Lawson's algorithm to find the triangle containing a point.
    Starts from the most recently added triangle and "walks" towards the point.
    Uses the adjacency information from triangle_neighbors to improve efficiency.

    Parameters:
    - all_points: Array of all point coordinates
    - triangle_vertices: Matrix where each row contains vertex indices of a triangle
    - triangle_neighbors: Matrix where each row contains adjacent triangle indices
    - point: The point to locate
    - last_triangle_idx: Index of the last formed triangle (starting point for search)

    Returns:
    - Index of the triangle containing the point, or None if not found
    """
    if triangulation.triangle_vertices.shape[0] == 0 or last_triangle_idx < 0:
        raise ValueError("No triangles available or invalid starting triangle")

    # Start from the last added triangle
    triangle_idx = last_triangle_idx

    # Keep track of visited triangles to avoid cycles
    visited = {triangle_idx}
    while True:
        logger.debug(f"visiting {triangle_idx}")

        # Get the current triangle vertices
        v_indices = triangulation.triangle_vertices[triangle_idx]
        triangle = triangulation.all_points[v_indices]

        # Check if the point is inside this triangle
        if point_inside_triangle(triangle, point):
            return triangle_idx

        # If not inside, find which edge to cross using the adjacent triangles information
        # Get edge indices in the triangle. NOTE!!! => it should be consistent with triangle_neighbors
        edges = [(1, 2), (2, 0), (0, 1)]
        next_idx = None
        for i, (e1, e2) in enumerate(edges):
            # Vector from edge to point
            edge_vector = triangle[e2] - triangle[e1]
            point_vector = point - triangle[e1]

            # If cross product is negative, the point is on the "outside" of this edge
            cross_prod = (
                edge_vector[0] * point_vector[1] - edge_vector[1] * point_vector[0]
            )
            if cross_prod < 0:
                # Get the adjacent triangle for this edge
                adjacent_idx = triangulation.triangle_neighbors[triangle_idx, i]

                # If there's an adjacent triangle (not a boundary) and we haven't visited it
                if adjacent_idx != -1 and adjacent_idx not in visited:
                    next_idx = adjacent_idx
                    visited.add(adjacent_idx)
                    break

        if next_idx is None:
            raise ValueError(f"Couldn't find a triangle containing {point}")
        triangle_idx = next_idx


def get_sorted_points(
    points: NDArray[np.floating], debug: bool = False
) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """
    Sort points into a spatially coherent order to improve incremental insertion efficiency.

    :param points: input points
    :param debug: if True, show a plot of the grid and points
    :return: sorted points and their original indices
    """

    # sort the points into bins
    grid_size = int(np.sqrt(len(points)))
    grid_size = max(grid_size, 4)  # Minimum grid size of 4x4

    y_idxs = (0.99 * grid_size * points[:, 1]).astype(int)
    x_idxs = (0.99 * grid_size * points[:, 0]).astype(int)

    # Create bin numbers in a snake-like pattern
    bin_numbers = np.zeros(len(points), dtype=int)
    for i in range(len(points)):
        y, x = y_idxs[i], x_idxs[i]
        if y % 2 == 0:
            bin_numbers[i] = y * grid_size + x
        else:
            bin_numbers[i] = (y + 1) * grid_size - x - 1

    # Sort the points by their bin numbers
    sorted_indices = np.argsort(bin_numbers)
    sorted_points = points[sorted_indices]

    if debug:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot the grid
        for i in range(grid_size + 1):
            ax.axhline(i / grid_size, color="gray", linestyle="--", linewidth=0.5)
            ax.axvline(i / grid_size, color="gray", linestyle="--", linewidth=0.5)

        # Plot the points
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c="blue",
            label="Original points",
        )
        ax.scatter(
            sorted_points[:, 0],
            sorted_points[:, 1],
            c="red",
            s=10,
            label="Sorted points",
            alpha=0.6,
        )

        ax.set_title("Debug: Normalized Points and Grid")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend()
        plt.grid(False)
        plt.show()

    return sorted_points, sorted_indices


def initialize_triangulation(
    sorted_points: NDArray[np.floating],
    margin: float = 10.0,
) -> Triangulation:
    """
    Initialize the triangulation with a super triangle.

    :param sorted_points: Sorted input points
    :param margin: extra margin to ensure all points are inside the super triangle
    :return: All points array, triangle vertices, triangle neighbors, and last triangle index
    """
    super_vertices = np.array(
        [
            [-margin + 0.5, -margin / 2],
            [margin + 0.5, -margin / 2],
            [0.5, margin],
        ]
    )

    # Add super-triangle vertices to the points array
    all_points = np.vstack([sorted_points, super_vertices])
    n_original_points = len(sorted_points)

    # Initial triangle is the super-triangle
    # Vertex indices are n_original_points, n_original_points+1, n_original_points+2
    triangle_vertices = np.array(
        [[n_original_points, n_original_points + 1, n_original_points + 2]]
    )

    # Initial triangle has no neighbors (all boundaries)
    triangle_neighbors = np.array([-1, -1, -1], dtype=int).reshape(1, -1)

    return Triangulation(
        all_points=all_points,
        triangle_vertices=triangle_vertices,
        triangle_neighbors=triangle_neighbors,
    )


def insert_point(
    point_idx: int,
    point: NDArray[np.floating],
    triangulation: Triangulation,
    debug: bool = False,
) -> Triangulation:
    """
    Insert a point into the triangulation.

    :param point_idx: Index of the point to insert
    :param point: Coordinates of the point
    :param triangulation:
    :param debug:
    :return: Updated triangle_vertices, triangle_neighbors, last_triangle_idx
    """
    # Find the triangle containing the point
    containing_idx = find_containing_triangle(
        triangulation, point, triangulation.last_triangle_idx
    )

    # Get vertices of the containing triangle
    v1_idx, v2_idx, v3_idx = triangulation.triangle_vertices[containing_idx]
    logger.debug(
        f"Triangle {containing_idx} with vertices {v1_idx}, {v2_idx}, {v3_idx} contains point {point_idx}"
    )

    # Get the original neighbors before we modify anything
    orig_neighbors = triangulation.triangle_neighbors[containing_idx].copy()
    neighbor_opp_v1 = orig_neighbors[0]  # neighbor opposite to v1 (across edge v2-v3)
    neighbor_opp_v2 = orig_neighbors[1]  # neighbor opposite to v2 (across edge v3-v1)
    neighbor_opp_v3 = orig_neighbors[2]  # neighbor opposite to v3 (across edge v1-v2)
    logger.debug(
        f"Neighbours: opposite v3={neighbor_opp_v3}, opposite v1={neighbor_opp_v1}, opposite v2={neighbor_opp_v2}"
    )

    # Create three new triangles by connecting point to each vertex
    # Note that we need to maintain CCW ordering for each triangle

    # Triangle 1: point + edge (v1, v2)
    original_t12_vertices = np.array([point_idx, v1_idx, v2_idx])
    new_triangle_12 = ensure_ccw_triangle(
        original_t12_vertices, triangulation.all_points
    )

    # Triangle 2: point + edge (v2, v3)
    original_t23_vertices = np.array([point_idx, v2_idx, v3_idx])
    new_triangle_23 = ensure_ccw_triangle(
        original_t23_vertices, triangulation.all_points
    )

    # Triangle 3: point + edge (v3, v1)
    original_t31_vertices = np.array([point_idx, v3_idx, v1_idx])
    new_triangle_31 = ensure_ccw_triangle(
        original_t31_vertices, triangulation.all_points
    )

    # Update triangulation data structure
    # first, update containing_idx
    triangulation.triangle_vertices[containing_idx] = new_triangle_12

    # Then add two more triangles
    triangulation.triangle_vertices = np.vstack(
        (
            triangulation.triangle_vertices,
            [new_triangle_23],
            [new_triangle_31],
        )
    )

    # Get indices for the new triangles
    triangle_count = len(triangulation.triangle_vertices)
    new_triangle_12_idx = containing_idx  # reusing the original index
    new_triangle_23_idx = triangle_count - 2
    new_triangle_31_idx = triangle_count - 1

    # Update neighbors array to accommodate new triangles
    # Add rows for the two new triangles
    triangulation.triangle_neighbors = np.vstack(
        (
            triangulation.triangle_neighbors,
            np.full((2, 3), -1, dtype=int),  # Initialize with -1 for no neighbor
        )
    )

    # Set up neighbor relationships for the three new triangles
    # for every triangle [v1, v2, v3] we define the neighbor as:
    # [t_sharing_edge_opposite_of_v1, t_sharing_edge_opposite_of_v2, t_sharing_edge_opposite_of_v3]

    # Triangle 1 (point, v1, v2)
    original_neighs_t12 = [
        neighbor_opp_v3,  # Edge (v1, v2) -> original neighbor opposite to v3
        new_triangle_23_idx,  # Edge (v2, point) -> triangle_23
        new_triangle_31_idx,  # Edge (point, v1) -> triangle_31
    ]
    triangulation.triangle_neighbors[new_triangle_12_idx] = (
        reorder_neighbors_for_triangle(
            original_t12_vertices,
            new_triangle_12,
            original_neighs_t12,
        )
    )

    # Triangle 2 (point, v2, v3)
    original_neighs_t23 = [
        neighbor_opp_v1,  # Edge (v2, v3) -> original neighbor opposite to v1
        new_triangle_31_idx,  # Edge (v3, point) -> triangle_31
        new_triangle_12_idx,  # Edge (point, v2) -> triangle_12
    ]
    triangulation.triangle_neighbors[new_triangle_23_idx] = (
        reorder_neighbors_for_triangle(
            original_t23_vertices,
            new_triangle_23,
            original_neighs_t23,
        )
    )

    # Triangle 3 (point, v3, v1)
    original_neighs_t31 = [
        neighbor_opp_v2,  # Edge (v3, v1) -> original neighbor opposite to v2
        new_triangle_12_idx,  # Edge (v1, point) -> triangle_12
        new_triangle_23_idx,  # Edge (point, v3) -> triangle_23
    ]
    triangulation.triangle_neighbors[new_triangle_31_idx] = (
        reorder_neighbors_for_triangle(
            original_t31_vertices,
            new_triangle_31,
            original_neighs_t31,
        )
    )

    # Update the original neighbors to point to the correct new triangles
    # and place them on the stack used for Lawson swapping
    stack = []

    def update_external_neighbor(neighbor_idx: int, new_triangle_idx: int):
        if neighbor_idx >= 0:
            stack.append((int(neighbor_idx), int(new_triangle_idx)))
            neighbor_refs = triangulation.triangle_neighbors[neighbor_idx]
            for i, ref in enumerate(neighbor_refs):
                if ref == containing_idx:
                    triangulation.triangle_neighbors[neighbor_idx, i] = new_triangle_idx
                    break

    # Update each external neighbor to point to the correct new triangle
    update_external_neighbor(neighbor_opp_v1, new_triangle_23_idx)  # v2-v3 edge
    update_external_neighbor(neighbor_opp_v2, new_triangle_31_idx)  # v3-v1 edge
    update_external_neighbor(neighbor_opp_v3, new_triangle_12_idx)  # v1-v2 edge

    # Restore Delaunay triangulation (edge flipping)
    if stack:
        lawson_swapping(point_idx, stack, triangulation)

    if debug:
        triangulation.plot()

    # Update last_triangle_idx to one of the new triangles
    triangulation.last_triangle_idx = triangle_count - 1
    return triangulation


def remove_super_triangle_triangles(
    triangulation: Triangulation,
    n_original_points: int,
) -> None:
    """
    Modify the Triangulation object in-place by removing triangles
    that contain vertices from the super triangle.

    :param triangulation: The Triangulation object to modify.
    :param n_original_points: Number of original points (before adding super triangle vertices).
    """
    mask = np.all(triangulation.triangle_vertices < n_original_points, axis=1)

    # remove triangles
    triangulation.triangle_vertices = triangulation.triangle_vertices[mask]
    # remove neighbors
    triangulation.triangle_neighbors = triangulation.triangle_neighbors[mask]
    triangulation.last_triangle_idx = triangulation.triangle_vertices.shape[0]
    # remove super-triangle points
    triangulation.all_points = triangulation.all_points[:-3]


def triangulate(
    points: NDArray[np.floating],
    margin: float = 10.0,
    debug: bool = False
):
    """
    Implement Delaunay triangulation using the incremental algorithm with efficient
    adjacency tracking.

    :param points: Input points to triangulate
    :param margin: initial margin for the supertriangle points
    :param debug: plot debug images
    :return: List of triangles forming the Delaunay triangulation
    """
    # Sort points for efficient insertion
    # Find the min and max along each axis (x and y)
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # Normalize the points to [0, 1] range
    normalized_points = (points - min_vals) / (max_vals - min_vals)
    # sorted_points, sorted_indices = get_sorted_points(normalized_points)  # TODO: reactivate?
    sorted_points = normalized_points

    # Initialize triangulation with super triangle
    triangulation = initialize_triangulation(sorted_points, margin=margin)

    n_original_points = len(sorted_points)

    # Loop over each point and insert into triangulation
    for point_idx, point in enumerate(sorted_points):
        # Insert point and update triangulation
        triangulation = insert_point(
            point_idx=point_idx,
            point=point,
            triangulation=triangulation,
            debug=debug,
        )

    # Remove triangles that contain vertices of the super triangle
    remove_super_triangle_triangles(triangulation, n_original_points)

    # apply inverse initial transformation. TODO: not needed, we only need to know how points are connected
    triangulation.all_points = (
        triangulation.all_points * (max_vals - min_vals) + min_vals
    )

    return triangulation


def remove_holes_old(
    triangulation: Triangulation, outer: list[int], holes: list[list[int]]
) -> None:
    """
    Get the final triangulation, compute the centroids for all the triangles. If a centroid is outside the domain,
    remove the corresponding triangle.
    TODO: caveats we can have 1) Centroid Inside, Triangle Outside, 2) Centroid Outside, Triangle Inside
    for a more robust version, use this strategy for all vertices (if a point is on the boundary, then it's inside)
    TODO: points in outer and holes should be sorted as those in triangulation do!
    """
    tri_vertices = triangulation.triangle_vertices
    to_delete = []
    poly_outer = [triangulation.all_points[p] for p in outer]
    poly_holes = []
    for hole in holes:
        poly_hole = [triangulation.all_points[p] for p in hole]
        poly_holes.append(poly_hole)

    for idx, row in enumerate(tri_vertices):
        tri_points = triangulation.all_points[row]
        centroid = np.mean(tri_points, axis=0)
        if not is_inside_domain(centroid, poly_outer, poly_holes):
            to_delete.append(idx)

    logger.info(f"Removing triangles {to_delete}")
    new_tri_vertices = [
        row for idx, row in enumerate(tri_vertices) if idx not in to_delete
    ]
    # TODO: update triangle neighbors?
    triangulation.triangle_vertices = new_tri_vertices  # type: ignore[reportAttributeAccessIssue]


def remove_holes(
    triangulation: Triangulation, outer: list[int], holes: list[list[int]]
) -> None:
    """
    Remove triangles outside the polygon (outer boundary with optional holes).
    Rebuilds triangle_neighbors to keep mesh topology consistent.
    """
    tri_vertices = triangulation.triangle_vertices
    all_points = triangulation.all_points
    to_delete = []

    # Convert vertex indices to coordinates
    poly_outer = [all_points[p] for p in outer]
    poly_holes = [[all_points[p] for p in hole] for hole in holes]

    # Classify triangles by centroid
    for idx, row in enumerate(tri_vertices):
        tri_points = all_points[row]
        centroid = np.mean(tri_points, axis=0)
        if not is_inside_domain(centroid, poly_outer, poly_holes):
            to_delete.append(idx)

    logger.info(f"Removing triangles {to_delete}")

    # Keep only triangles not marked for deletion
    keep_mask = np.ones(len(tri_vertices), dtype=bool)
    keep_mask[to_delete] = False
    new_tri_vertices = tri_vertices[keep_mask]

    # --- Rebuild neighbors ---
    edge_to_triangle = {}
    for t_idx, tri in enumerate(new_tri_vertices):
        for i in range(3):
            v1, v2 = tri[(i + 1) % 3], tri[(i + 2) % 3]  # edge opposite vertex i
            edge_to_triangle[(v1, v2)] = t_idx

    new_neighbors = np.full((len(new_tri_vertices), 3), -1, dtype=int)
    for t_idx, tri in enumerate(new_tri_vertices):
        for i in range(3):
            v1, v2 = tri[(i + 1) % 3], tri[(i + 2) % 3]
            if (v2, v1) in edge_to_triangle:
                new_neighbors[t_idx, i] = edge_to_triangle[(v2, v1)]

    triangulation.triangle_vertices = new_tri_vertices
    triangulation.triangle_neighbors = new_neighbors
    triangulation.last_triangle_idx = len(new_tri_vertices) - 1
