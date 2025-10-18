import os
import cv2
import numpy as np

# =============== Настройки ===============
# Если хочешь явно указать путь к видео — укажи здесь:
VIDEO_PATH = ""   # например: "sample.mp4" или "data/video.avi"
ALLOWED_IMG_EXT = (".jpg", ".jpeg", ".png")
# ========================================

def find_image_label_dirs():
    """
    Ищет папки images/labels в двух распространённых вариантах структуры.
    Возвращает (images_dir, labels_dir).
    """
    candidates = [
        os.path.join("datasets", "vpapenko", "nails-segmentation", "versions", "1", "nails_segmentation"),
        os.path.join("datasets", "vpapenko", "nails-segmentation", "versions", "1"),
    ]
    for base in candidates:
        images = os.path.join(base, "images")
        labels = os.path.join(base, "labels")
        if os.path.isdir(images) and os.path.isdir(labels):
            return images, labels
    raise FileNotFoundError("Не нашёл папки images/labels. Проверь структуру проекта.")

def collect_pairs(images_dir, labels_dir):
    """
    Собирает пары (image_path, mask_path) по совпадающим именам без расширения.
    """
    imgs = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(ALLOWED_IMG_EXT)])
    msks = sorted([f for f in os.listdir(labels_dir) if f.lower().endswith(ALLOWED_IMG_EXT)])

    img_map = {os.path.splitext(f)[0]: f for f in imgs}
    msk_map = {os.path.splitext(f)[0]: f for f in msks}

    common = sorted(set(img_map) & set(msk_map))
    pairs = [(os.path.join(images_dir, img_map[k]), os.path.join(labels_dir, msk_map[k])) for k in common]

    only_img = sorted(set(img_map) - set(msk_map))
    only_msk = sorted(set(msk_map) - set(img_map))
    return pairs, only_img, only_msk

def contours_on_image(image_bgr, mask_img):
    """
    Вычисляет контуры на маске и рисует их поверх изображения.
    Маска может быть одно- или трёхканальная.
    Возвращает: (img_with_contours, mask_vis)
      - img_with_contours: исходник с зелёными контурами
      - mask_vis: цветная визуализация маски (красная заливка)
    """
    # Приводим маску к одному каналу
    if len(mask_img.shape) == 3:
        mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_img

    # Бинаризация: всё >0 считаем маской
    _, bin_mask = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)

    # Контуры
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Подгон размеров, если не совпадают
    if image_bgr.shape[:2] != bin_mask.shape[:2]:
        bin_mask = cv2.resize(bin_mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        # пересчёт контуров после ресайза (простой способ — найти снова)
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Рисуем контуры
    img_contours = image_bgr.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)  # зелёные линии

    # Цветная визуализация маски
    mask_vis = np.zeros_like(image_bgr)
    mask_vis[bin_mask > 0] = (0, 0, 255)  # красный

    return img_contours, mask_vis

def show_pairs_viewer(pairs):
    """
    Просмотрщик пар: слева — исходник с контурами, справа — цветная маска.
    Управление:
      ←/A — назад, →/D — вперёд, Q/Esc — выход.
    """
    print(f"Найдено пар: {len(pairs)}")
    print("Управление: ←/A — назад, →/D — вперёд, Q/Esc — выход")

    i = 0
    win = "Nails: image + contours | mask"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        img_path, msk_path = pairs[i]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        msk = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)

        if img is None or msk is None:
            print(f"⚠️ Пропуск повреждённого файла:\n  {img_path}\n  {msk_path}")
            i = (i + 1) % len(pairs)
            continue

        img_with_cnt, mask_vis = contours_on_image(img, msk)

        # Подгоняем маску к размеру изображения на всякий случай
        if img_with_cnt.shape[:2] != mask_vis.shape[:2]:
            mask_vis = cv2.resize(mask_vis, (img_with_cnt.shape[1], img_with_cnt.shape[0]), interpolation=cv2.INTER_NEAREST)

        side = np.hstack([img_with_cnt, mask_vis])

        title = f"[{i+1}/{len(pairs)}] img: {os.path.basename(img_path)} | mask: {os.path.basename(msk_path)}"
        vis = side.copy()
        cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow(win, vis)
        key = cv2.waitKey(0) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (81, ord('a'), ord('A')):   # ← / A
            i = (i - 1) % len(pairs)
        elif key in (83, ord('d'), ord('D')):   # → / D
            i = (i + 1) % len(pairs)

    cv2.destroyWindow(win)

def find_any_video_path():
    """
    Если VIDEO_PATH не задан, пытается найти любой .mp4/.avi/.mov в текущей папке.
    """
    if VIDEO_PATH and os.path.isfile(VIDEO_PATH):
        return VIDEO_PATH
    for f in os.listdir("."):
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            return f
    return ""

def play_video_gray(video_path):
    """
    Проигрывает видеофайл в градациях серого.
    Управление: Q/Esc — выход.
    """
    if not video_path:
        print("Видео не найдено. Укажи путь в VIDEO_PATH или положи файл рядом со скриптом.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        return

    win = f"Video (grayscale): {os.path.basename(video_path)}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("Проигрывание видео в оттенках серого. Нажми Q/Esc для выхода.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow(win, gray)
        key = cv2.waitKey(25) & 0xFF  # ~40 fps -> 25ms
        if key in (27, ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyWindow(win)

def main():
    # 1) Пары изображение–маска
    images_dir, labels_dir = find_image_label_dirs()
    pairs, only_img, only_msk = collect_pairs(images_dir, labels_dir)

    if not pairs:
        print("Пар не найдено.")
        if only_img:
            print("Есть изображения без масок (первые 10):", only_img[:10])
        if only_msk:
            print("Есть маски без изображений (первые 10):", only_msk[:10])
    else:
        show_pairs_viewer(pairs)

    # 2) Видео в градациях серого
    video = find_any_video_path()
    play_video_gray(video)

if __name__ == "__main__":
    main()
