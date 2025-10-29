import cv2
import numpy as np
import time

VIDEO_SOURCE = 0      # 0 = webcam,
RED_SEC   = 3.0       # detection ON
GREEN_SEC = 3.0       # detection OFF

BLUR_K = (7, 7)
DIFF_THR = 25
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

FLOW = dict(pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

def open_capture(src):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {src}")
    return cap

def to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def motion_mask_diff(prev_gray, gray):
    # Basic: |curr - prev| -> threshold -> de-noise
    g1 = cv2.GaussianBlur(prev_gray, BLUR_K, 0)
    g2 = cv2.GaussianBlur(gray,     BLUR_K, 0)
    diff = cv2.absdiff(g1, g2)
    _, m = cv2.threshold(diff, DIFF_THR, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL, iterations=1)
    return m

def motion_mask_flow(prev_gray, gray):
    # Robust: optical flow magnitude -> threshold
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **FLOW)
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=False)
    m = (mag > 1.5).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL, iterations=1)
    return m

def two_color_map(mask, shape_hw):
    # Red where motion, green elsewhere
    h, w = shape_hw
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:] = (0, 255, 0)
    out[mask > 0] = (0, 0, 255)
    return out

def main():
    cap = open_capture(VIDEO_SOURCE)
    use_flow = False               # False=Diff, True=Flow
    phase = None
    next_switch = time.time() + GREEN_SEC
    prev_gray = None

    win_cam = "Camera"
    win_map = "Motion Map"
    cv2.namedWindow(win_cam, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_map, cv2.WINDOW_NORMAL)


    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = to_gray(frame)

        # Phase timer
        now = time.time()
        if now >= next_switch:
            if phase == "GREEN":
                phase = "RED"
                next_switch = now + RED_SEC
            else:
                phase = "GREEN"
                next_switch = now + GREEN_SEC

        # Build motion map ALWAYS (red where motion, green elsewhere)
        # Build motion map (only when RED phase is active)
        if phase == "RED" and prev_gray is not None:
            if use_flow:
                mask = motion_mask_flow(prev_gray, gray)  # Optical Flow (Farneback)
            else:
                mask = motion_mask_diff(prev_gray, gray)  # Frame difference
            map_img = two_color_map(mask, gray.shape)
        else:
            # During GREEN phase or first frame: all green (no motion detection)
            map_img = two_color_map(np.zeros_like(gray), gray.shape)

        # Text at edge (top-left)
        mode = "Flow" if use_flow else "Diff"
        label = "RED light (detect ON)" if phase == "RED" else "GREEN light (detect OFF)"
        color = (0,0,255) if phase == "RED" else (0,255,0)
        cv2.putText(frame, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Mode: {mode}", (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # Show
        cv2.imshow(win_cam, frame)
        cv2.imshow(win_map, map_img)
        prev_gray = gray

        # Keys
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('o'), ord('O')):
            use_flow = not use_flow

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
