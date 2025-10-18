# main.py
import os
from pathlib import Path
import csv
import cv2
import numpy as np

# --- приглушаем болтовню OpenCV ---
if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

# ====== НАСТРОЙКИ ======
# Укажи базу датасета (папка versions/1)
BASE = Path(r"C:/Users/salap/OneDrive/Рабочий стол/Учеба/Питон/PAC/kagglehub/datasets/vpapenko/nails-segmentation/versions/1")

# Возможные расположения (некоторые сборки имеют вложенную папку nails_segmentation)
CANDIDATES = [
    (BASE / "images", BASE / "labels"),
    (BASE / "nails_segmentation" / "images", BASE / "nails_segmentation" / "labels"),
]

# Сохранять предпросмотры вместо показа окон (True/False)
SAVE_PREVIEWS = True
PREVIEW_DIR = BASE / "preview"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

# Сохранить pairs.csv (True/False)
SAVE_CSV = True
PAIRS_CSV = BASE / "pairs.csv"
# =======================


def choose_dirs(candidates):
    """Берём первую пару (images, labels), где есть файлы."""
    for imgs, labs in candidates:
        if imgs.exists() and labs.exists():
            try:
                has_imgs = any(p.is_file() for p in imgs.iterdir())
                has_labs = any(p.is_file() for p in labs.iterdir())
            except PermissionError:
                has_imgs = has_labs = False
            if has_imgs and has_labs:
                return imgs, labs
    raise RuntimeError("Не найдено валидных папок images/labels. Проверь структуру датасета.")


def index_by_stem(folder: Path):
    """Строим индекс {stem: [Path,...]} с поддержкой разных расширений."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    idx = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            idx.setdefault(p.stem, []).append(p)
    return idx


def read_image_any(path: Path, flags=cv2.IMREAD_COLOR):
    """
    Надёжное чтение: открываем файл в бинарном режиме и декодируем через imdecode.
    Это обходит лимит длинных путей и выносок OneDrive.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден (возможно OneDrive-placeholder): {path}")
    size = path.stat().st_size
    if size == 0:
        raise IOError(f"Файл 0 байт (не скачан из OneDrive): {path}")
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise IOError(f"Не удалось декодировать изображение: {path}")
    return img


def make_preview(img_bgr: np.ndarray, mask_gray: np.ndarray) -> np.ndarray:
    """Красиво накладываем маску на изображение."""
    if mask_gray.ndim != 2:
        raise ValueError(f"Маска должна быть 2D, получено shape={mask_gray.shape}")
    # бинаризуем на случай не 0/255
    m = (mask_gray > 127).astype(np.uint8) * 255
    color_mask = cv2.applyColorMap(m, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img_bgr, 0.7, color_mask, 0.3, 0)
    return blended


def main():
    images_dir, labels_dir = choose_dirs(CANDIDATES)
    print(f"Используем:\n  images: {images_dir}\n  labels: {labels_dir}")

    imgs = index_by_stem(images_dir)
    labs = index_by_stem(labels_dir)

    pairs = []
    missing = []
    for stem, img_list in imgs.items():
        if stem in labs:
            pairs.append((img_list[0], labs[stem][0]))
        else:
            missing.append(stem)

    print(f"Всего найдено пар: {len(pairs)}")
    if missing:
        print(f"⚠ Нет масок для {len(missing)} файлов (первые 5): {missing[:5]}")

    # Проверка размеров первых файлов (видно, скачаны ли они локально)
    for p_img, p_msk in pairs[:3]:
        try:
            print("CHECK:", p_img, "size=", p_img.stat().st_size, "|", p_msk, "size=", p_msk.stat().st_size)
        except Exception as e:
            print("CHECK error:", e)

    # Опционально сохраним pairs.csv
    if SAVE_CSV:
        with open(PAIRS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "mask"])
            for ip, mp in pairs:
                w.writerow([str(ip), str(mp)])
        print(f"Список пар сохранён: {PAIRS_CSV}")

    # Визуализация / предпросмотры
    saved = 0
    shown = 0
    for img_path, msk_path in pairs:
        try:
            img = read_image_any(img_path, cv2.IMREAD_COLOR)
            mask = read_image_any(msk_path, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            print(f"SKIP: {img_path.name} — {e}")
            continue

        preview = make_preview(img, mask)

        if SAVE_PREVIEWS:
            out = PREVIEW_DIR / f"{img_path.stem}_preview.jpg"
            # cv2.imwrite сам справится, путь короткий
            cv2.imwrite(str(out), preview)
            saved += 1
        else:
            cv2.imshow("image + mask", preview)
            if cv2.waitKey(250) & 0xFF == 27:
                break
            shown += 1
            if shown >= 20:
                break

    if not SAVE_PREVIEWS:
        cv2.destroyAllWindows()
    else:
        print(f"Предпросмотры сохранены в: {PREVIEW_DIR.resolve()} ({saved} шт.)")


if __name__ == "__main__":
    main()
