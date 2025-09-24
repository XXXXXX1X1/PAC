def analyze(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    num_lines = len(text.splitlines())
    num_words = len(text.split())
    num_letters = sum(1 for letter in text if letter.isalpha())

    return num_lines, num_words, num_letters


lines, words, letters = analyze("input.txt")
print("Строк:", lines)
print("Слов:", words)
print("Букв:", letters)
