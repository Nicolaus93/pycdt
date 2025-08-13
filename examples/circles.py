import numpy as np

from src.delaunay import remove_holes, triangulate


def generate_concentric_circles(center=(0, 0), num_points=20, debug=False):
    import matplotlib.pyplot as plt

    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = []
    radii = (0.5, 1.0)
    colors = ["red", "blue"]

    for r, color in zip(radii, colors):
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        circle_points = np.stack((x, y), axis=1)  # Shape: (num_points, 2)
        points.append(circle_points)

        if debug:
            plt.scatter(x, y, label=f"r = {r}", color=color)
            for i, (xi, yi, angle) in enumerate(zip(x, y, theta)):
                # Offset the label slightly outward from the circle
                offset_radius = 0.05
                offset_x = xi + offset_radius * np.cos(angle)
                offset_y = yi + offset_radius * np.sin(angle)
                plt.text(
                    offset_x, offset_y, str(i), fontsize=8, ha="center", va="center"
                )

    if debug:
        plt.gca().set_aspect("equal")
        plt.title("Concentric Circle Points")
        plt.legend()
        plt.grid(True)
        plt.show()

    # return np.vstack(points)  # Shape: (num_points * len(radii), 2)
    return points


def triangle_concentric_circles():
    n = 20
    inner_circle, outer_circle = generate_concentric_circles(num_points=n, debug=True)
    yy = triangulate(np.vstack([outer_circle, inner_circle]), debug=True)
    yy.export_animation_matplotlib("../assets/triangulation.gif")
    yy.plot()
    remove_holes(yy, [i for i in range(n)], [[i for i in range(n, 2 * n)]])
    yy.plot(exclude_super_t=True)


if __name__ == "__main__":
    triangle_concentric_circles()
