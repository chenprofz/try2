import math

def group_points_with_group_id(points, square_size, max_squares):
    min_x = min(points, key=lambda p: p[0])[0]
    max_x = max(points, key=lambda p: p[0])[0]
    min_y = min(points, key=lambda p: p[1])[1]
    max_y = max(points, key=lambda p: p[1])[1]

    width = max_x - min_x
    height = max_y - min_y

    num_columns = math.ceil(width / square_size)
    num_rows = math.ceil(height / square_size)

    best_solution = []

    def group_points_recursive(squares, square_index, remaining_points):
        nonlocal best_solution

        if square_index == max_squares:
            if len(remaining_points) == 0:
                if len(squares) < len(best_solution):
                    best_solution = squares.copy()
            return

        for i in range(square_index, num_rows * num_columns):
            square = get_square_from_index(i)
            if not is_square_intersecting(square, squares):
                new_squares = squares[:]
                new_squares.append(square)

                new_remaining_points = remaining_points[:]
                for point in remaining_points:
                    if is_point_in_square(point, square):
                        new_remaining_points.remove(point)

                group_points_recursive(new_squares, square_index + 1, new_remaining_points)

    def get_square_from_index(index):
        column = index % num_columns
        row = index // num_columns

        x = min_x + column * square_size
        y = min_y + row * square_size

        square = (x, y, square_size)
        return square

    def is_square_intersecting(square, squares):
        for existing_square in squares:
            if check_square_intersection(square, existing_square):
                return True
        return False

    def check_square_intersection(square1, square2):
        x1, y1, size1 = square1
        x2, y2, size2 = square2

        if abs(x1 - x2) < size1 + size2 and abs(y1 - y2) < size1 + size2:
            return True
        return False

    def is_point_in_square(point, square):
        x, y, size = square
        point_x, point_y = point

        if abs(point_x - x) <= size / 2 and abs(point_y - y) <= size / 2:
            return True
        return False

    group_points_recursive([], 0, points)

    groups = {}
    for i, square in enumerate(best_solution):
        for point in points:
            if is_point_in_square(point, square):
                if point not in groups:
                    groups[point] = []

                groups[point].append(i)

    return groups

# Example usage:
points = [(1, 1), (2, 2), (4, 3), (3, 4), (5, 5), (6, 7), (9, 8)]
square_size = 3
max_squares = 3

result = group_points_with_group_id(points, square_size, max_squares)

for point, group_ids in result.items():
    print("Point:", point)
    print("Group IDs:", group_ids)
    print()