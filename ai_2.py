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
for (x, y), var in zip(rectangles, variables):
    if var.solution_value():
        squares.append((x, y))

# Plot the squares and points
fig, ax = plt.subplots()
for square in squares:
    ax.add_patch(Rectangle(square, w, h, alpha=0.5, edgecolor='red', facecolor='none'))
x, y = zip(*points)
plt.scatter(x, y, color='blue', label='Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Squares and Points')
plt.legend()
plt.show()