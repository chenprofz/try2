import math
import numpy as np
from scipy.spatial import cKDTree

def group_points_with_knn_squares(points, K, square_size):
    min_x = min(points, key=lambda p: p[0])[0]
    max_x = max(points, key=lambda p: p[0])[0]
    min_y = min(points, key=lambda p: p[1])[1]
    max_y = max(points, key=lambda p: p[1])[1]

    width = max_x - min_x
    height = max_y - min_y

    num_columns = math.ceil(width / square_size)
    num_rows = math.ceil(height / square_size)

    squares = np.empty((num_rows, num_columns), dtype=object)

    for point in points:
        x = point[0]
        y = point[1]

        column = math.floor((x - min_x) / square_size)
        row = math.floor((y - min_y) / square_size)

        if squares[row, column] is None:
            squares[row, column] = []

        squares[row, column].append(point)

    groups = {}

    for row in range(num_rows):
        for column in range(num_columns):
            if squares[row, column] is not None:
                for point in squares[row, column]:
                    neighbors = []
                    for i in range(max(0, row-1), min(num_rows, row+2)):
                        for j in range(max(0, column-1), min(num_columns, column+2)):
                            if squares[i, j] is not None:
                                neighbors.extend(squares[i, j])

                    kdtree = cKDTree(neighbors)
                    _, indices = kdtree.query(point, K+1)
                    indices = indices[1:]  # Exclude the point itself from the neighbors

                    if len(indices) < K:
                        groups[point] = neighbors
                    else:
                        groups[point] = [neighbors[i] for i in indices]

    return groups

# Example usage:
points = [(1, 1), (3, 4), (2, 2), (4, 3), (5, 5)]
K = 2
square_size = 2

result = group_points_with_knn_squares(points, K
