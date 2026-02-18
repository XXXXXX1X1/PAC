#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np


def read_array(path: str) -> np.ndarray:
    """Считывает числа из файла в numpy-массив."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip().replace('\n', ' ')
    return np.fromstring(text, sep=' ', dtype=int)


def method1(real, synth, P, rng):
    """1 способ — через np.where"""
    mask = rng.random(real.shape) < P
    return np.where(mask, synth, real)


def method2(real, synth, P, rng):
    """2 способ — через np.choose"""
    idx = rng.choice(2, size=real.shape, p=[1 - P, P])
    return np.choose(idx, [real, synth])


def method3(real, synth, P, rng):
    """3 способ — через перестановку"""
    n = real.size
    k = int(round(P * n))
    mask = np.zeros(n, dtype=bool)
    mask[rng.permutation(n)[:k]] = True
    return np.where(mask, synth, real)


def main():
    # создаём парсер аргументов
    parser = argparse.ArgumentParser(
        description="Смешивает два массива по вероятности P тремя способами."
    )
    parser.add_argument("real", help="Файл с реальными числами")
    parser.add_argument("synth", help="Файл с синтетическими числами")
    parser.add_argument("P", type=float, help="Вероятность выбора из synth (0..1)")
    parser.add_argument("--seed", type=int, default=None, help="Сид генератора случайных чисел")

    args = parser.parse_args()

    try:
        # чтение данных
        real = read_array(args.real)
        synth = read_array(args.synth)

        # проверка совпадения размеров
        if real.shape != synth.shape:
            print("Ошибка: файлы содержат разное количество чисел!", file=sys.stderr)
            sys.exit(1)

        rng = np.random.default_rng(args.seed)

        # три способа
        r1 = method1(real, synth, args.P, rng)
        r2 = method2(real, synth, args.P, rng)
        r3 = method3(real, synth, args.P, rng)

        print("Метод 1 (np.where):")
        print(" ".join(map(str, r1)))
        print("\nМетод 2 (np.choose):")
        print(" ".join(map(str, r2)))
        print("\nМетод 3 (перестановка):")
        print(" ".join(map(str, r3)))

    except FileNotFoundError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Ошибка выполнения: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
