import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import math

def generate_random_points(n):
    # Generate n random points using make_blobs
    X, _ = make_blobs(n_samples=n, centers=3, random_state=0)
    points = [tuple(point) for point in X]
    return points

def group_points_with_knn(points, k, max_iterations):
    kmeans = KMeans(n_clusters=k, max_iter=max_iterations, random_state=0)
    kmeans.fit(points)
    group_assignments = kmeans.labels_
    groups = {}

    for i, point in enumerate(points):
        group_id = group_assignments[i]
        if group_id not in groups:
            groups[group_id] = []

        groups[group_id].append(point)

    return groups

def calculate_best_k(points, max_k):
    distortions = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(points)
        distortions.append(kmeans.inertia_)

    distortions = np.array(distortions)
    gradients = np.gradient(distortions)
    elbow_index = np.argmax(gradients)
    best_k = elbow_index + 1

    return best_k

def find_fixed_perfect_squares(points, square_size):
    min_x = min(x for x, _ in points)
    min_y = min(y for _, y in points)
    max_x = max(x for x, _ in points)
    max_y = max(y for _, y in points)

    # Calculate the number of perfect squares required
    num_squares_x = math.ceil((max_x - min_x) / square_size)
    num_squares_y = math.ceil((max_y - min_y) / square_size)
    num_squares = max(num_squares_x, num_squares_y)

    start_x = min_x - (min_x % square_size)
    start_y = min_y - (min_y % square_size)

    end_x = start_x + (num_squares * square_size)
    end_y = start_y + (num_squares * square_size)

    return (start_x, start_y), (end_x, end_y)

def plot_squares(grouped_points, squares):
    # Get unique group labels
    group_labels = set(grouped_points.keys())

    # Plot the points with different colors for each group
    for group_label in group_labels:
        group_points = grouped_points[group_label]
        x, y = zip(*group_points)
        plt.scatter(x, y, label=f'Group {group_label}')

    # Plot the squares
    for square in squares:
        start_point, end_point = square
        start_x, start_y = start_point
        end_x, end_y = end_point
        square_patch = plt.Rectangle((start_x, start_y), end_x - start_x, end_y - start_y, edgecolor='red', facecolor='none')
        plt.gca().add_patch(square_patch)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Points with Fixed-Size Squares')
    plt.legend()
    plt.show()

# Example usage:
num_points = 100
max_k = 10
max_iterations = 100
square_size = 5

# Generate random points
points = generate_random_points(num_points)

# Group the points using KNN
grouped_points = group_points_with_knn(points, 3, max_iterations)

# Concatenate all the points in all the groups
grouped_points = [point for group_points in grouped_points.values() for point in group_points]

# Determine the best K value based on the elbow method
best_k = calculate_best_k(grouped_points, max_k)

# Group the points using the best K value
grouped_points = group_points_with_knn(points, best_k, max_iterations)

# Concatenate all the points in all the groups
grouped_points = [point for group_points in grouped_points.values() for point in group_points]

# Find the fixed-size squares to cover all the points
squares = []
while len(grouped_points) > best_k:
    square = find_fixed_perfect_squares(grouped_points, square_size)
    squares.append(square)
    # Remove the points covered by the square
    grouped_points = [point for point in grouped_points if not (square[0][0] <= point[0] <= square[1][0] and square[0][1] <= point[1] <= square[1][1])]

# Plot the points with different colors based on their group and the fixed-size squares
plot_squares(grouped_points, squares)