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

squares = []
for i, (x, y) in enumerate(rectangles):
    if variables[i].solution_value():
        squares.append(((x, y), i))

# Plot the squares and points
fig, ax = plt.subplots()
for square, square_id in squares:
    (sx, sy) = square
    ax.add_patch(Rectangle((sx, sy), w, h, alpha=0.5, edgecolor='red', facecolor='none'))
    plt.text(sx + w / 2, sy + h / 2, str(square_id), color='red', fontsize=8, ha='center', va='center')
x, y = zip(*points)
plt.scatter(x, y, color='blue', label='Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Squares and Points with Square IDs')
plt.legend()
plt.show()