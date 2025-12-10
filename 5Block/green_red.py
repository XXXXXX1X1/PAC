import cv2
import numpy as np
import time

# Время длительности фаз "красный" и "зелёный"
RED_SEC   = 5.0
GREEN_SEC = 5.0

# Параметры фильтрации и порога
BLUR_K = (7, 7)   # размер ядра Гаусса
DIFF_THR = 25     # порог разницы для бинаризации
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # ядро для морф. операций


def to_gray(frame):
    """Преобразует кадр из BGR в оттенки серого."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def motion_mask_diff(prev_gray, gray):
    """Находит движение как разницу между текущим и предыдущим кадром."""
    g1 = cv2.GaussianBlur(prev_gray, BLUR_K, 0)  # сглаживание шума
    g2 = cv2.GaussianBlur(gray,     BLUR_K, 0)
    diff = cv2.absdiff(g1, g2)                   # |текущий - предыдущий|
    _, m = cv2.threshold(diff, DIFF_THR, 255, cv2.THRESH_BINARY)  # бинаризация
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL, iterations=1) # удаление мелких точек
    return m

def motion_mask_flow(prev_gray, gray):
    """Находит движение по оптическому потоку (Фарнебэк) с предварительным сглаживанием."""
    pg = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    cg = cv2.GaussianBlur(gray,      (5, 5), 0)
    #  Оптический поток
    flow = cv2.calcOpticalFlowFarneback(
        pg, cg, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    # Модуль вектора движения
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Порог по скорости → бинарная маска
    m = (mag > 2.0).astype(np.uint8)    * 255
    # Очистка мелкого шума
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL, iterations=1)
    return m



def two_color_map(mask):

    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:] = (0, 255, 0)        # зелёный фон (BGR)
    out[mask > 0] = (0, 0, 255) # красный для движущихся областей
    return out


def main():

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видеопоток с камеры")

    use_flow = False               # False=Diff, True=Flow
    phase = None                   # текущая фаза ("RED"/"GREEN")
    next_switch = time.time() + GREEN_SEC
    prev_gray = None

    cv2.namedWindow("Camera + Motion Map", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = to_gray(frame)
        now = time.time()


        if now >= next_switch:
            if phase == "GREEN":
                phase = "RED"
                next_switch = now + RED_SEC
            else:
                phase = "GREEN"
                next_switch = now + GREEN_SEC


        if phase == "RED" and prev_gray is not None:
            if use_flow:
                mask = motion_mask_flow(prev_gray, gray)
            else:
                mask = motion_mask_diff(prev_gray, gray)
            map_img = two_color_map(mask)
        else:
            map_img = two_color_map(np.zeros_like(gray))


        mode = "Flow" if use_flow else "Diff"
        label = "RED light (detect ON)" if phase == "RED" else "GREEN light (detect OFF)"
        color = (0,0,255) if phase == "RED" else (0,255,0)
        cv2.putText(frame, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Mode: {mode}", (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)


        combined = np.hstack([frame, map_img])
        cv2.imshow("Camera + Motion Map", combined)

        prev_gray = gray

        key = cv2.waitKey(1)
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('o'), ord('O')):
            use_flow = not use_flow

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
