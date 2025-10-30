import cv2
import os
import numpy as np

images_dir = "/Users/xxx/Desktop/Учеба/Python/Pac/5Block/datasets/vpapenko/nails-segmentation/versions/1/images"
labels_dir = "/Users/xxx/Desktop/Учеба/Python/Pac/5Block/datasets/vpapenko/nails-segmentation/versions/1/labels"


image_files = os.listdir(images_dir)
label_files = os.listdir(labels_dir)

# Создаём пары с одинаковым базовым именем
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


cv2.namedWindow('Nails Segmentation', cv2.WINDOW_NORMAL)

current_index = 0
while True:
    img_file, label_file = pairs[current_index]

    image_path = os.path.join(images_dir, img_file)
    mask_path  = os.path.join(labels_dir, label_file)

    image = cv2.imread(image_path)                       # BGR изображение
    mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # одно канал (0..255)

    if image is None or mask is None:
        print(f"Ошибка чтения {img_file} или {label_file}")
        current_index = (current_index + 1) % len(pairs)
        continue

    # Бинаризация маски: >127 -> 255, иначе 0
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)


    # Поиск контуров на бинарной маске
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Рисуем контуры зелёным на копии исходного изображения
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

    # Для показа маску переведём в BGR и соберём 3 кадра рядом
    mask_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    display = np.hstack([image, mask_bgr, image_with_contours])

    cv2.imshow('Nails Segmentation', display)

    # Клавиши: q — выход, a — назад, d — вперёд
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        current_index = (current_index - 1) % len(pairs)
    elif key == ord('d'):
        current_index = (current_index + 1) % len(pairs)

cv2.destroyAllWindows()
