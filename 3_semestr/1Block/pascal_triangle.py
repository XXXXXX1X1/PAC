import argparse

def pascal(n):
    triangle = [[1]]  # первая строка

    for i in range(1, n):
        prev = triangle[-1]
        row = [1]
        for j in range(len(prev) - 1):
            row.append(prev[j] + prev[j + 1])
        row.append(1)
        triangle.append(row)

    width = len(" ".join(map(str, triangle[-1])))
    for row in triangle:
        line = " ".join(map(str, row))
        print(line.center(width))

parser = argparse.ArgumentParser()
parser.add_argument("rows", type=int,)
args = parser.parse_args()

pascal(args.rows)

## python3 pascal_triangle.py 5
