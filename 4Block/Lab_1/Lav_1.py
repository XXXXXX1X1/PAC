import sys
import numpy as np


def read_array(path):
    """Считывает числа из файла в numpy-массив"""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip().replace('\n', ' ')
        return np.fromstring(text, sep=' ', dtype=int)


def method1(real, synth, P, rng):
    """1 способ — через np.where"""
    mask = rng.random(real.shape) < P
    return np.where(mask, synth, real)


def method2(real, synth, P, rng):
    """2 способ — через np.choose"""
    idx = rng.choice(2, size=real.shape, p=[1-P, P])  # 0 — real, 1 — synth
    return np.choose(idx, [real, synth])


def method3(real, synth, P, rng):
    """3 способ — через np.stack и продвинутую индексацию"""
    picks = (rng.random(real.shape) < P).astype(int)
    stacked = np.stack([real, synth])  # shape (2, n)
    return stacked[picks, np.arange(real.size)]


def main():
    if len(sys.argv) != 4:
        print("Использование: python3 random_select.py real.txt synth.txt P")
        sys.exit(1)

    real_file, synth_file, P_str = sys.argv[1], sys.argv[2], sys.argv[3]

    # читаем данные
    real = read_array(real_file)
    synth = read_array(synth_file)
    P = float(P_str)

    # проверки
    if real.shape != synth.shape:
        print("Ошибка: файлы содержат массивы разной длины!")
        sys.exit(2)
    if not (0 <= P <= 1):
        print("Ошибка: вероятность P должна быть от 0 до 1!")
        sys.exit(3)

    rng = np.random.default_rng()  # генератор случайных чисел

    # три способа перемешивания
    r1 = method1(real, synth, P, rng)
    r2 = method2(real, synth, P, rng)
    r3 = method3(real, synth, P, rng)

    # выводим три результата
    print(" ".join(map(str, r1)))
    print(" ".join(map(str, r2)))
    print(" ".join(map(str, r3)))


if __name__ == "__main__":
    main()

#python Lav_1.py text1.txt text2.txt 0.3
#python3 Lav_1.py text1.txt text2.txt 0.3