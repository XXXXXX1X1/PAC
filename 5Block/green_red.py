
# green_red.py (Unicode-ready)
import cv2
import numpy as np
import time
import os
from PIL import ImageFont, ImageDraw, Image

# -------------------- НАСТРОЙКИ --------------------
VIDEO_SOURCE = 0            # 0 = веб-камера; либо путь к файлу: 'video.mp4'
RED_SECONDS   = 3.0         # "Красный": детекция ВКЛ
GREEN_SECONDS = 3.0         # "Зелёный": детекция ВЫКЛ

# Детектор по разности кадров
BLUR_KSIZE = (7, 7)
DIFF_THRESH = 25
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
MOTION_RATIO_THRESH = 0.003

# Оптический поток Farnebäck
FLOW_WIN_SIZE = 15
FLOW_LEVELS   = 3
FLOW_ITER     = 3
FLOW_POLY_N   = 5
FLOW_POLY_SIG = 1.2
FLOW_MAG_THRESH = 1.5

# Путь к шрифту, поддерживающему кириллицу
# Для macOS часто подходит этот:
DEFAULT_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "arial.ttf",
]
FONT_CACHE = {}

# ---------------------------------------------------

def open_capture(src):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть источник видео: {src}")
    return cap

def to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

def diff_motion_mask(prev_gray, gray):
    g1 = cv2.GaussianBlur(prev_gray, BLUR_KSIZE, 0)
    g2 = cv2.GaussianBlur(gray,      BLUR_KSIZE, 0)
    diff = cv2.absdiff(g1, g2)
    _, mask = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, MORPH_KERNEL, iterations=1)
    return mask

def flow_motion_mask(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        pyr_scale=0.5, levels=FLOW_LEVELS, winsize=FLOW_WIN_SIZE,
                                        iterations=FLOW_ITER, poly_n=FLOW_POLY_N, poly_sigma=FLOW_POLY_SIG,
                                        flags=0)
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=False)
    mask = (mag > FLOW_MAG_THRESH).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)
    return mask

def two_color_map(mask, shape):
    h, w = shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:] = (0, 255, 0)
    out[mask > 0] = (0, 0, 255)
    return out

def motion_ratio(mask):
    return float(cv2.countNonZero(mask)) / (mask.shape[0] * mask.shape[1])

def tint_frame(frame, color_bgr, alpha):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (0,0), (w,h), color_bgr, -1)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

# -------- Unicode-отрисовка текста через Pillow --------
def get_font(size):
    key = (tuple(DEFAULT_FONTS), size)
    if key in FONT_CACHE:
        return FONT_CACHE[key]
    font = None
    for path in DEFAULT_FONTS:
        if os.path.isfile(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()  # на крайний случай (ASCII)
    FONT_CACHE[key] = font
    return font

def draw_text_center(img_bgr, text, y_rel=0.55, size=60,
                     color=(255,255,255), stroke=3, stroke_color=(0,0,0)):
    """Крупный центрированный Unicode-текст с корректным выравниванием."""
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    font = get_font(size)

    # получаем размеры текста (ширина, высота)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # координаты центра
    x = int((w - tw) / 2)
    y = int(h * y_rel - th / 2)

    # отрисовка обводки
    if stroke > 0:
        draw.text((x, y), text, font=font,
                  fill=stroke_color, stroke_width=stroke, stroke_fill=stroke_color)
    # основной текст
    draw.text((x, y), text, font=font, fill=color)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_text_tl(img_bgr, text, pos=(20, 40), size=36, color=(255,255,255), stroke=2, stroke_color=(0,0,0)):
    """Верхний левый Unicode-текст."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    font = get_font(size)
    if stroke > 0:
        draw.text(pos, text, font=font, fill=stroke_color, stroke_width=stroke, stroke_fill=stroke_color)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
# -------------------------------------------------------

def choose_video_path():
    print("Укажи путь к видеофайлу (mp4/avi/mov/mkv): ", end="", flush=True)
    path = input().strip()
    if path and os.path.isfile(path):
        return path
    print("Файл не найден — остаюсь на текущем источнике.")
    return None

def main():
    global VIDEO_SOURCE
    cap = open_capture(VIDEO_SOURCE)

    use_flow = False
    phase = "GREEN"
    next_switch = time.time() + GREEN_SECONDS

    win_cam = "GreenRed — камера"
    win_map = "GreenRed — карта"
    cv2.namedWindow(win_cam, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_map, cv2.WINDOW_NORMAL)

    prev_gray = None
    print("Управление: Q/Esc — выход | O — OpticalFlow on/off | G — выбрать видеофайл | Space — смена фазы")

    while True:
        ok, frame = cap.read()
        if not ok:
            if isinstance(VIDEO_SOURCE, str):
                cap.release()
                cap = open_capture(VIDEO_SOURCE)
                prev_gray = None
                continue
            else:
                break

        gray = to_gray(frame)

        # Переключение фаз
        now = time.time()
        if now >= next_switch:
            if phase == "GREEN":
                phase = "RED"
                next_switch = now + RED_SECONDS
            else:
                phase = "GREEN"
                next_switch = now + GREEN_SECONDS
        remaining = max(0, int(next_switch - now))

        show = frame.copy()

        if phase == "RED" and prev_gray is not None:
            mask_diff = diff_motion_mask(prev_gray, gray)
            if use_flow:
                mask_flow = flow_motion_mask(prev_gray, gray)
                mask = cv2.bitwise_or(mask_diff, mask_flow)
            else:
                mask = mask_diff

            move = motion_ratio(mask) > MOTION_RATIO_THRESH
            map_img = two_color_map(mask if move else np.zeros_like(mask), frame.shape)

            show = tint_frame(show, (0, 0, 60), 0.6)
            show = draw_text_center(show, "КРАСНЫЙ СВЕТ", y_rel=0.60, size=64, color=(255,0,0), stroke=6, stroke_color=(0,0,0))
            if move:
                show = draw_text_center(show, "ПОЙМАН!", y_rel=0.38, size=72, color=(255,0,0), stroke=7, stroke_color=(0,0,0))
            show = draw_text_tl(show, f"Осталось: {remaining}s", pos=(20, 40), size=36)
            mode = "Flow" if use_flow else "Diff"
            show = draw_text_tl(show, f"Det: {mode} | ratio>{MOTION_RATIO_THRESH}", pos=(20, frame.shape[0]-50), size=28)
        else:
            map_img = two_color_map(np.zeros(gray.shape, np.uint8), frame.shape)
            show = tint_frame(show, (0, 60, 0), 0.4)
            show = draw_text_center(show, "ЗЕЛЁНЫЙ СВЕТ", y_rel=0.60, size=64, color=(0,255,0), stroke=6, stroke_color=(0,0,0))
            show = draw_text_tl(show, f"До КРАСНОГО: {remaining}s", pos=(20, 40), size=36)

        cv2.imshow(win_cam, show)
        cv2.imshow(win_map, map_img)
        prev_gray = gray

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('o'), ord('O')):
            use_flow = not use_flow
        elif key in (ord('g'), ord('G')):
            path = choose_video_path()
            if path:
                VIDEO_SOURCE = path
                cap.release()
                cap = open_capture(VIDEO_SOURCE)
                prev_gray = None
        elif key == 32:  # Space
            if phase == "GREEN":
                phase = "RED";   next_switch = time.time() + RED_SECONDS
            else:
                phase = "GREEN"; next_switch = time.time() + GREEN_SECONDS

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
