import cv2

def main():
    # 🔹 Укажи путь к видеофайлу здесь
    video_path = "/Users/xxx/Desktop/Учеба/Python/Pac/5Block/sample-10s.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        return

    cv2.namedWindow("Grayscale Video", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Grayscale Video", gray)

        key = cv2.waitKey(10) & 0xFF
        if key in (ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
