class Item:
    def __init__(self, ripe, count=1, max_count=32, color='green', saturation=10):
        self._ripe = ripe
        self._count = count
        self._max_count = max_count
        self._color = color
        self._saturation = saturation

    def update_count(self, val):
        if 0 <= val <= self._max_count:
            self._count = val
            return True
        return False

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, val):
        if 0 <= val <= self._max_count:
            self._count = val

    @staticmethod
    def static():
        print('I am function')

    @classmethod
    def my_name(cls):
        return cls.__name__

    @property
    def color(self):
        return self._color

    @property
    def eatable(self):
        return self._ripe

    def __add__(self, other):
        new_count = self._count + other
        if 0 <= new_count <= self._max_count:
            return Item(self._ripe, new_count, self._max_count, self._color, self._saturation)
        return False

    def __sub__(self, other):
        new_count = self._count - other
        if 0 <= new_count <= self._max_count:
            return Item(self._ripe, new_count, self._max_count, self._color, self._saturation)
        return False

    def __mul__(self, other):
        new_count = self._count * other
        if 0 <= new_count <= self._max_count:
            return Item(self._ripe, new_count, self._max_count, self._color, self._saturation)
        return False

    def __iadd__(self, other):
        new_count = self._count + other
        if 0 <= new_count <= self._max_count:
            self._count = new_count
            return self
        return False

    def __isub__(self, other):
        new_count = self._count - other
        if 0 <= new_count <= self._max_count:
            self._count = new_count
            return self
        return False

    def __imul__(self, other):
        new_count = self._count * other
        if 0 <= new_count <= self._max_count:
            self._count = new_count
            return self
        return False

    def __lt__(self, other):
        return self._count < other

    def __gt__(self, other):
        return self._count > other

    def __le__(self, other):
        return self._count <= other

    def __ge__(self, other):
        return self._count >= other

    def __eq__(self, other):
        return self._count == other

    def __len__(self):
        return self._count

class Pineapple(Item):
    def __init__(self, ripe=True, count=1, max_count=32, color='yellow', saturation=7):
        super().__init__(ripe, count, max_count, color, saturation)

class Melon(Item):
    def __init__(self, ripe=True, count=1, max_count=32, color='green', saturation=10):
        super().__init__(ripe, count, max_count, color, saturation)

class Potato(Item):
    def __init__(self, ripe=True, count=1, max_count=100, color='brown', saturation=5):
        super().__init__(ripe, count, max_count, color, saturation)

class Carrot(Item):
    def __init__(self, ripe=True, count=1, max_count=50, color='orange', saturation=6):
        super().__init__(ripe, count, max_count, color, saturation)

class Inventory:
    def __init__(self, size):
        self._size = size
        self._items = [None] * size

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        if value is None or value.eatable:
            self._items[index] = value
        else:
            raise ValueError("Можно добавлять только съедобные объекты или None")

    def decrease(self, index, amount=1):
        item = self._items[index]
        if item is None:
            return
        item.count -= amount
        if item.count <= 0:
            self._items[index] = None

    def __str__(self):
        return str([f"{item.__class__.__name__}({item.count})" if item else None for item in self._items])

    item1 = Item(ripe=True, count=1, max_count=10, color='red')
    print(item1)
    # inv = Inventory(3)               #init
    # pi = Pineapple(True, count=5)
    # po = Potato(True, count=10)
    # inv[0] = pi                      #setitem
    # inv[1] = po
    # print(pi.count > 4)
    #
    # print(inv[0])
    #
    # print(inv)  #['Pineapple(5)', 'Potato(10)', None]       #str
    #
    # inv.decrease(0, 2)
    # inv.decrease(1, 5)
    # print(inv)  #['Pineapple(3)', 'Potato(5)', None]
    #
    # inv.decrease(0, 3)
    # print(inv)  #[None, 'Potato(5)', None]