import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

# Example usage:
points = [(1, 1), (2, 2), (4, 3), (3, 4), (5, 5), (6, 7), (9, 8)]
k = 3
max_iterations = 100

result = group_points_with_knn(points, k, max_iterations)

# Plotting the grouped points
colors = ['red', 'green', 'blue']
for group_id, group_points in result.items():
    group_color = colors[group_id]
    x, y = zip(*group_points)
    plt.scatter(x, y, color=group_color, label=f'Group {group_id}')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Grouped Points')
plt.legend()
plt.show()