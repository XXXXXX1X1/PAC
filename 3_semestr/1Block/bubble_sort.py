import argparse

def bubble_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

parser = argparse.ArgumentParser()
parser.add_argument("numbers", nargs="+", type=int)
args = parser.parse_args()

print("Исходный список:", args.numbers)
print("Отсортированный список:", bubble_sort(args.numbers))

##python3 bubble_sort.py 5 2 9 1 3

