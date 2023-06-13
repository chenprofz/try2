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

def plot_elbow_curve(points, max_k):
    distortions = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(points)
        distortions.append(kmeans.inertia_)

    # Plotting the elbow curve
    plt.plot(range(1, max_k + 1), distortions, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Distortion')
    plt.title('Elbow Curve')
    plt.show()

# Example usage:
num_points = 100
max_k = 10
max_iterations = 100

# Generate random points
points = generate_random_points(num_points)

# Plot the elbow curve
plot_elbow_curve(points, max_k)

# Determine the best K based on the elbow curve
best_k = int(input("Enter the best K value based on the elbow curve: "))

# Group the points using the best K
result = group_points_with_knn(points, best_k, max_iterations)

# Plotting the grouped points
colors = ['red', 'green', 'blue', 'orange', 'purple']
for group_id, group_points in result.items():
    group_color = colors[group_id]
    x, y = zip(*group_points)
    plt.scatter(x, y, color=group_color, label=f'Group {group_id}')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Grouped Points')
plt.legend()
plt.show()