import argparse

# Преобразует текстовый блок в матрицу
def parse_matrix(block):
    rows = []  # список строк матрицы
    for line in block.strip().splitlines():
        row = [int(x) for x in line.split()]  # одна строка матрицы
        rows.append(row)
    width = len(rows[0])  # количество столбцов
    if any(len(r) != width for r in rows):
        raise ValueError("Матрица не прямоугольная")
    return rows

# Читает из файла две матрицы
def read_two_matrices(path):
    with open(path, "r") as f:
        content = f.read()
    blocks = [b for b in content.split("\n\n") if b]  # две части файла
    if len(blocks) != 2:
        raise ValueError("В файле должно быть ровно две матрицы")
    return parse_matrix(blocks[0]), parse_matrix(blocks[1])

# Умножает матрицы A и B
def multiply_matrices(A, B):
    n, m = len(A), len(A[0])   # размеры A
    m2, k = len(B), len(B[0])  # размеры B
    if m != m2:
        raise ValueError("Размерности не согласованы")
    C = [[0] * k for i in range(n)]  # результирующая матрица
    for i in range(n):       # по строкам A
        for t in range(m):   # по столбцам A / строкам B
            for j in range(k):  # по столбцам B
                C[i][j] += A[i][t] * B[t][j]
    return C

# Записывает матрицу в файл
def write_matrix(matrix, path):
    with open(path, "w") as f:
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    A, B = read_two_matrices(args.input)
    C = multiply_matrices(A, B)
    write_matrix(C, args.output)

if __name__ == "__main__":
    main()

##python3 matrix_mult.py matrix.txt result.txt