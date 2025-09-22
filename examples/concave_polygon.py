import numpy as np

from src.build import triangulate


if __name__ == "__main__":
    points = [
        (0, 0),
        (4, 0),
        (4, 4),
        (3, 4),
        (3, 1),
        (1, 1),
        (1, 4),
        (0, 4),
    ]

    arr = np.array(points)
    yy = triangulate(arr)
    yy.plot(show=True)
