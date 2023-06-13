import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

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

def find_fixed_size_squares(points):
    squares = []
    remaining_points = set(points)

    while remaining_points:
        current_point = remaining_points.pop()
        x, y = current_point

        # Calculate the starting coordinates for the square
        square_size = math.ceil(math.sqrt(len(remaining_points)))  # Calculate the size as square root of remaining points rounded up
        start_x = x - (x % square_size)
        start_y = y - (y % square_size)

        # Add the square to the list
        squares.append((start_x, start_y, square_size))

        # Remove points covered by the square
        points_to_remove = []
        for point in remaining_points:
            px, py = point
            if start_x <= px < start_x + square_size and start_y <= py < start_y + square_size:
                points_to_remove.append(point)

        for point in points_to_remove:
            remaining_points.remove(point)

    return squares

def plot_squares(points, squares, square_size):
    # Plot the points
    x, y = zip(*points)
    plt.scatter(x, y, color='blue', label='Points')

    # Plot the squares
    for square in squares:
        start_x, start_y = square
        square_patch = plt.Rectangle((start_x, start_y), square_size, square_size, edgecolor='red', facecolor='none')
        plt.gca().add_patch(square_patch)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Minimum Squares to Cover Points')
    plt.legend()
    plt.show()

# Example usage:
num_points = 100
best_k = 3
square_size = 4

# Generate random points
points = generate_random_points(num_points)

# Group the points using KNN with the best K value
grouped_points = group_points_with_knn(points, best_k, max_iterations=100)

# Concatenate all the points in all the groups
grouped_points = [point for group_points in grouped_points.values() for point in group_points]

# Find the minimum squares to cover all the points
squares = find_fixed_size_squares(grouped_points, square_size)

# Plot the squares and points
plot_squares(grouped_points, squares, square_size)