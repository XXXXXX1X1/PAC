import cv2

video_path = "sample-10s.mp4"  # путь к видеофайлу

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Не удалось открыть видео: {video_path}")
    exit()

cv2.namedWindow("Grayscale Video", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Video", gray)

    key = cv2.waitKey(10)
    if key in (27, ord('q')):  # Esc или q — выход
        break

cap.release()
cv2.destroyAllWindows()
