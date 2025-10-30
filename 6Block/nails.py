import os, random, cv2, numpy as np

# ---------- загрузка ----------
def _load_pairs(images_dir, masks_dir):
    # ТВОЙ ПОИСК ПАР (двойной цикл по спискам файлов)
    image_files = [f for f in os.listdir(images_dir) if not f.startswith('.')]
    label_files = [f for f in os.listdir(masks_dir)  if not f.startswith('.')]

    pairs = []
    for img_file in image_files:
        name = os.path.splitext(img_file)[0]
        for label_file in label_files:
            label_name = os.path.splitext(label_file)[0]
            if name == label_name:
                # возвращаем ПОЛНЫЕ ПУТИ, чтобы ниже ничего не менять
                pairs.append((os.path.join(images_dir, img_file),
                              os.path.join(masks_dir,  label_file)))
                break

    if not pairs:
        raise RuntimeError("Не найдено пар изображение-маска.")
    return pairs

def _read_img_mask(img_path, msk_path):
    img = cv2.imread(img_path)
    msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
    return img, msk

def _resize_pair(img, msk, size):
    h, w = size
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    msk = cv2.resize(msk, (w, h), interpolation=cv2.INTER_NEAREST)
    return img, msk

# ---------- аугментации ----------
def _rand_rotate(img, msk, max_angle=180):
    angle = random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT101)
    msk = cv2.warpAffine(msk, M, (w, h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, msk, f"rotate={angle:.1f}°"

def _rand_flip(img, msk):
    ops = []
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        msk = cv2.flip(msk, 1)
        ops.append("flipH")
    if random.random() < 0.5:
        img = cv2.flip(img, 0)
        msk = cv2.flip(msk, 0)
        ops.append("flipV")
    return img, msk, ops

def _rand_crop(img, msk, min_scale=0.6):
    h, w = img.shape[:2]
    scale = random.uniform(min_scale, 1.0)
    ch, cw = int(h * scale), int(w * scale)
    top, left = random.randint(0, h - ch), random.randint(0, w - cw)
    img = img[top:top+ch, left:left+cw]
    msk = msk[top:top+ch, left:left+cw]
    img = cv2.resize(img, (w, h))
    msk = cv2.resize(msk, (w, h), interpolation=cv2.INTER_NEAREST)
    return img, msk, f"crop={int(scale*100)}%"

def _rand_blur(img):
    if random.random() < 0.5:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
        return img, f"blur{k}x{k}"
    return img, None

def augment_pair(img, msk):
    ops = []
    if random.random() < 0.5:
        img, msk, op = _rand_rotate(img, msk)
        ops.append(op)
    img, msk, flips = _rand_flip(img, msk)
    ops.extend(flips)
    if random.random() < 0.5:
        img, msk, op = _rand_crop(img, msk)
        ops.append(op)
    img, op = _rand_blur(img)
    if op: ops.append(op)
    if not ops:
        ops.append("none")
    return img, msk, ", ".join(ops)

# ---------- генератор ----------
def nails_generator(images_dir, masks_dir, target_size=(256,256)):
    pairs = _load_pairs(images_dir, masks_dir)
    while True:
        img_path, msk_path = random.choice(pairs)
        img, msk = _read_img_mask(img_path, msk_path)
        img_orig, msk_orig = _resize_pair(img, msk, target_size)
        img_aug, msk_aug, info = augment_pair(img_orig.copy(), msk_orig.copy())
        yield img_orig, msk_orig, img_aug, msk_aug, info

# ---------- визуализация ----------
if __name__ == "__main__":
    images_dir = "/Users/xxx/Desktop/Учеба/Python/Pac/5Block/datasets/vpapenko/nails-segmentation/versions/1/images"
    masks_dir  = "/Users/xxx/Desktop/Учеба/Python/Pac/5Block/datasets/vpapenko/nails-segmentation/versions/1/labels"
    gen = nails_generator(images_dir, masks_dir, target_size=(256,256))

    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Preview", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
    cv2.resizeWindow("Preview", 1000, 800)
    print("Пробел — новая комбинация, Q — выход")

    for img_orig, msk_orig, img_aug, msk_aug, info in gen:
        color_orig = cv2.applyColorMap(msk_orig, cv2.COLORMAP_JET)
        color_aug  = cv2.applyColorMap(msk_aug,  cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_aug, 0.7, color_aug, 0.3, 0)

        cv2.putText(overlay, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)

        row1 = np.hstack([img_orig, color_orig])
        row2 = np.hstack([img_aug, overlay])
        view = np.vstack([row1, row2])

        cv2.imshow("Preview", view)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
