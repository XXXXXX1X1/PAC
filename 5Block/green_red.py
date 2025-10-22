import os
import cv2
import numpy as np

# --- Чёткий путь к данным ---
BASE_DIR = os.path.join("datasets", "vpapenko", "nails-segmentation", "versions", "1")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
MASKS_DIR = os.path.join(BASE_DIR, "labels")

IMG_EXTS = {".jpg", ".jpeg", ".png"}
MSK_EXTS = {".png", ".jpg", ".jpeg"}

def list_files_by_stem(folder, allowed_exts):
    files = {}
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in allowed_exts:
            stem = os.path.splitext(name)[0]
            files[stem] = os.path.join(folder, name)
    return files

def make_pairs(images_dir, masks_dir):
    imgs = list_files_by_stem(images_dir, IMG_EXTS)
    msks = list_files_by_stem(masks_dir, MSK_EXTS)
    common = sorted(set(imgs.keys()) & set(msks.keys()))
    return [(imgs[k], msks[k]) for k in common]

def load_and_draw(img_path, mask_path):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        raise FileNotFoundError("Не удалось загрузить одно из изображений.")

    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Бинаризация
    _, bin_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # Морфологическая очистка от одиночных пикселей
    kernel = np.ones((5, 5), np.uint8)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)   # удаляет мелкие шумы
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)  # заполняет мелкие дырки

    # Поиск контуров
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Рисуем контуры на копии изображения
    vis = img.copy()
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    return np.hstack([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), vis])


def main():
    pairs = make_pairs(IMAGES_DIR, MASKS_DIR)
    if not pairs:
        print("Пар не найдено.")
        return

    idx = 0
    cv2.namedWindow("Nails Segmentation Viewer", cv2.WINDOW_NORMAL)

    while True:
        img_path, mask_path = pairs[idx]
        panel = load_and_draw(img_path, mask_path)
        cv2.imshow("Nails Segmentation Viewer", panel)

        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):
            break
        elif key in (ord('n'), ord('d')):  # → или d
            idx = (idx + 1) % len(pairs)
        elif key in (ord('p'), ord('a')):  # ← или a
            idx = (idx - 1) % len(pairs)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
