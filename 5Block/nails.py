import cv2
import os
import numpy as np

images_dir = "archive/images"
labels_dir = "archive/labels"

# Получаем списки файлов
image_files = [f for f in os.listdir(images_dir)]
label_files = [f for f in os.listdir(labels_dir)]

# Создаем пары по одинаковым именам
pairs = []
for img_file in image_files:
    name = os.path.splitext(img_file)[0]
    for label_file in label_files:
        label_name = os.path.splitext(label_file)[0]
        if name == label_name:
            pairs.append((img_file, label_file))
            break

if not pairs:
    print("Не найдено пар изображение-маска")
    exit()

# Создаем одно окно с именем
cv2.namedWindow('Nails Segmentation', cv2.WINDOW_NORMAL)

current_index = 0
while True:
    img_file, label_file = pairs[current_index]

    # Объединяем пути и названия файлов
    image_path = os.path.join(images_dir, img_file)
    mask_path = os.path.join(labels_dir, label_file)

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Ошибка чтения {img_file} или {label_file}")
        current_index = (current_index + 1) % len(pairs)
        continue

    # Бинаризация маски
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Поиск контуров
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Рисуем контуры на изображении (зелёный цвет)
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

    # Собираем всё в один вид
    mask_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    display = np.hstack([image, mask_bgr, image_with_contours])

    # Обновляем окно с правильным именем
    cv2.imshow('Nails Segmentation', display)

    # Обработка клавиш
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):  # выход
        break
    elif key == ord('a'):  # назад
        current_index = (current_index - 1) % len(pairs)
    elif key == ord('d'):  # вперёд
        current_index = (current_index + 1) % len(pairs)

cv2.destroyAllWindows()
print("Завершено!")
