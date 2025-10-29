import os
import cv2
import numpy as np
import argparse

ALLOWED_IMG_EXT = (".jpg", ".jpeg", ".png")

# ----------------------- аргументы -----------------------

def parse_args():
    p = argparse.ArgumentParser(description="Просмотр ногтей или видео в градациях серого.")
    p.add_argument("--video", action="store_true", help="Если указан, воспроизводится только видео.")
    p.add_argument("--video-path", type=str, default="", help="Путь к видеофайлу (например sample.mp4).")
    return p.parse_args()

# ----------------------- поиск директорий -----------------------

def find_image_label_dirs():
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

# ----------------------- сбор пар -----------------------

def collect_pairs(images_dir, labels_dir):
    imgs = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(ALLOWED_IMG_EXT)])
    msks = sorted([f for f in os.listdir(labels_dir) if f.lower().endswith(ALLOWED_IMG_EXT)])
    img_map = {os.path.splitext(f)[0]: f for f in imgs}
    msk_map = {os.path.splitext(f)[0]: f for f in msks}
    common = sorted(set(img_map) & set(msk_map))
    pairs = [(os.path.join(images_dir, img_map[k]), os.path.join(labels_dir, msk_map[k])) for k in common]
    return pairs

# ----------------------- контуры -----------------------

def contours_on_image(image_bgr, mask_img):
    if len(mask_img.shape) == 3:
        mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_img
    _, bin_mask = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)
    if image_bgr.shape[:2] != bin_mask.shape[:2]:
        bin_mask = cv2.resize(bin_mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = image_bgr.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    mask_vis = np.zeros_like(image_bgr)
    mask_vis[bin_mask > 0] = (0, 0, 255)
    return img_contours, mask_vis

# ----------------------- просмотр пар -----------------------

def show_pairs_viewer(pairs):
    print(f"Найдено пар: {len(pairs)}")
    print("←/A — назад, →/D — вперёд, Q/Esc — выход")
    i = 0
    win = "Nails: image + contours | mask"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        img_path, msk_path = pairs[i]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        msk = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)
        if img is None or msk is None:
            i = (i + 1) % len(pairs)
            continue
        img_with_cnt, mask_vis = contours_on_image(img, msk)
        side = np.hstack([img_with_cnt, mask_vis])
        title = f"[{i+1}/{len(pairs)}] {os.path.basename(img_path)}"
        cv2.putText(side, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow(win, side)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (81, ord('a'), ord('A')):
            i = (i - 1) % len(pairs)
        elif key in (83, ord('d'), ord('D')):
            i = (i + 1) % len(pairs)
    cv2.destroyWindow(win)


# ----------------------- main -----------------------

def main():
    args = parse_args()

             # показываем только ногти
    images_dir, labels_dir = find_image_label_dirs()
    pairs = collect_pairs(images_dir, labels_dir)
    if not pairs:
        print("Пар не найдено.")
    else:
        show_pairs_viewer(pairs)

if __name__ == "__main__":
    main()
