import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ortools.linear_solver import pywraplp

w = 0.1
h = 0.1
points = [(random.random(), random.random()) for _ in range(100)]
rectangles = [(x, y) for (x, _) in points for (_, y) in points]
solver = pywraplp.Solver.CreateSolver("min cover", "SCIP")
objective = solver.Objective()
constraints = [solver.RowConstraint(1, pywraplp.inf, str(p)) for p in points]
variables = [solver.BoolVar(str(r)) for r in rectangles]
for (x, y), var in zip(rectangles, variables):
    objective.SetCoefficient(var, 1)
    for (px, py), con in zip(points, constraints):
        if x <= px <= x + w and y <= py <= y + h:
            con.SetCoefficient(var, 1)
solver.Objective().SetMinimization()
solver.Solve()

squares = {}
point_squares = {}  # Dictionary to store the square ID for each point
for i, (x, y) in enumerate(rectangles):
    if variables[i].solution_value():
        if i not in squares:
            squares[i] = []
        squares[i].append((x, y))
        for j, point in enumerate(points):
            if x <= point[0] <= x + w and y <= point[1] <= y + h:
                point_squares[j] = i

# Plot the squares and points
fig, ax = plt.subplots()
for square_id, square_points in squares.items():
    (sx, sy) = square_points[0]
    ax.add_patch(Rectangle((sx, sy), w, h, alpha=0.5, edgecolor='red', facecolor='none'))
    plt.text(sx + w / 2, sy + h / 2, str(square_id), color='red', fontsize=8, ha='center', va='center')
    plt.plot(sx + w / 2, sy + h / 2, 'ro', markersize=3)
    for point in square_points:
        px, py = point
        plt.text(px, py, str(square_id), color='black', fontsize=6, ha='center', va='center')

for i, (x, y) in enumerate(points):
    plt.text(x, y, str(i), color='blue', fontsize=6, ha='center', va='center')
    square_id = point_squares.get(i)
    if square_id is not None:
        (sx, sy) = squares[square_id][0]
        plt.plot([x, sx + w / 2], [y, sy + h / 2], '--', color='gray')

x, y = zip(*points)
plt.scatter(x, y, color='blue', label='Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Squares and Points with Square IDs')
plt.legend()
plt.show()
