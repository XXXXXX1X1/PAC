#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import sys
from typing import List, Tuple

Matrix = List[List[int]]

def parse_two_matrices(path: str) -> Tuple[Matrix, Matrix]:
    """
    Читает файл и возвращает две целочисленные матрицы, разделённые пустой строкой.
    Проверяет прямоугольность (одинаковое число столбцов в каждой строке).
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Разбиваем на два блока по пустой строке(ам)
    blocks = re.split(r"\n\s*\n", text)
    if len(blocks) != 2:
        raise ValueError(
            f"Ожидалось ровно две матрицы, разделённые пустой строкой, а получено блоков: {len(blocks)}"
        )

    def parse_block(block: str) -> Matrix:
        rows = []
        for line in block.strip().splitlines():
            if not line.strip():
                continue
            nums = line.strip().split()
            try:
                row = [int(x) for x in nums]
            except ValueError as e:
                raise ValueError(f"Невозможно преобразовать в int: строка «{line}»") from e
            rows.append(row)
        if not rows:
            raise ValueError("Пустая матрица.")
        # Проверим прямоугольность
        w = len(rows[0])
        if any(len(r) != w for r in rows):
            raise ValueError("Матрица не прямоугольная (разное число столбцов в строках).")
        return rows

    A = parse_block(blocks[0])
    K = parse_block(blocks[1])
    return A, K

def flip_kernel(kernel: Matrix) -> Matrix:
    """Переворот ядра по вертикали и горизонтали (для настоящей свёртки)."""
    return [row[::-1] for row in kernel[::-1]]

def conv2d_valid(image: Matrix, kernel: Matrix, flip: bool = True) -> Matrix:
    """
    Свёртка (valid) без паддинга, шаг 1.
    Если flip=True — настоящая свёртка (ядро переворачивается).
    Если flip=False — корреляция (ядро не переворачивается).
    """
    n = len(image)
    m = len(image[0])
    r = len(kernel)
    c = len(kernel[0])

    if r > n or c > m:
        raise ValueError(
            f"Ядро больше исходной матрицы: image={n}x{m}, kernel={r}x{c}"
        )

    K = flip_kernel(kernel) if flip else kernel

    out_h = n - r + 1
    out_w = m - c + 1
    out = [[0 for _ in range(out_w)] for _ in range(out_h)]

    for i in range(out_h):
        for j in range(out_w):
            s = 0
            # суммирование по ядру
            for u in range(r):
                # локальная ссылка для скорости
                img_row = image[i + u]
                ker_row = K[u]
                for v in range(c):
                    s += img_row[j + v] * ker_row[v]
            out[i][j] = s
    return out

def write_matrix(path: str, M: Matrix) -> None:
    """Записывает матрицу в файл: строки — по одной строке, числа через пробел."""
    with open(path, "w", encoding="utf-8") as f:
        for row in M:
            f.write(" ".join(str(x) for x in row) + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="ЛР 2.2: valid-свёртка (или корреляция) двух матриц из файла."
    )
    parser.add_argument("input", help="Путь к входному файлу с двумя матрицами, разделёнными пустой строкой.")
    parser.add_argument("output", help="Путь к выходному файлу для записи результатной матрицы.")
    parser.add_argument(
        "--mode",
        choices=["conv", "corr"],
        default="conv",
        help="Режим: conv — свёртка (ядро переворачивается), corr — корреляция (без переворота). По умолчанию: conv.",
    )
    args = parser.parse_args()

    try:
        A, K = parse_two_matrices(args.input)
    except Exception as e:
        print(f"[Ошибка чтения] {e}", file=sys.stderr)
        sys.exit(1)

    flip = (args.mode == "conv")
    try:
        C = conv2d_valid(A, K, flip=flip)
    except Exception as e:
        print(f"[Ошибка вычисления] {e}", file=sys.stderr)
        sys.exit(2)

    try:
        write_matrix(args.output, C)
    except Exception as e:
        print(f"[Ошибка записи] {e}", file=sys.stderr)
        sys.exit(3)

    # Небольшая подсказка в консоль
    print(f"Готово. Размеры: A={len(A)}x{len(A[0])}, K={len(K)}x{len(K[0])} → Out={len(C)}x{len(C[0])}")
    print(f"Режим: {'свёртка (conv)' if flip else 'корреляция (corr)'}")
    print(f"Результат записан в: {args.output}")

if __name__ == "__main__":
    main()
