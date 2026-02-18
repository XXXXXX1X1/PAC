class Lupa:
    def __init__(self, name):
        self.name = name
        self.salary = 0

    def take_salary(self, count):
        self.salary += count

    def do_work(self, file1, file2):
        A = []
        B = []

        with open(file1, "r") as f:
            for line in f:
                row = [int(x) for x in line.split()]
                A.append(row)

        with open(file2, "r") as f:
            for line in f:
                row = [int(x) for x in line.split()]
                B.append(row)

        result = []
        for i in range(len(A)):
            row = []
            for j in range(len(A[0])):
                row.append(A[i][j] - B[i][j])
            result.append(row)

        print(f"результат работы {self.name}: ")
        for row in result:
            print(row)


class Pupa:
    def __init__(self, name):
        self.name = name
        self.salary = 0

    def take_salary(self, count):
        self.salary += count

    def do_work(self, file1, file2):
        A = []
        B = []

        with open(file1, "r") as f:
            for line in f:
                row = [int(x) for x in line.split()]
                A.append(row)

        with open(file2, "r") as f:
            for line in f:
                row = [int(x) for x in line.split()]
                B.append(row)

        result = []
        for i in range(len(A)):
            row = []
            for j in range(len(A[0])):
                row.append(A[i][j] + B[i][j])
            result.append(row)

        print(f"результат работы {self.name}: ")
        for row in result:
            print(row)


class Accountant:
    def __init__(self, name):
        self.name = name

    def give_salary(self, worker, amount):
        worker.take_salary(amount)
        print(f"{worker.name} получил {amount}")

if __name__ == "__main__":
    pupa = Pupa("Pupa")
    lupa = Lupa("Lupa")
    acc = Accountant("Галя")

    # Даем зарплату
    acc.give_salary(pupa, 666)
    acc.give_salary(lupa, 1)


    pupa.do_work("matrix1.txt", "matrix2.txt")
    print()
    lupa.do_work("matrix1.txt", "matrix2.txt")