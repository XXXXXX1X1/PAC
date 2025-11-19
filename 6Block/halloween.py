import sys, os, cv2, numpy as np

RATIO = 0.75
RANSAC_THR = 5.0
MIN_INLIERS = 12
COLOR = (0, 255, 0)
THICK = 3
IOU_MERGE = 0.4

def bbox(poly):
    xs = poly[:,0,0]; ys = poly[:,0,1]
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)

def iou(a, b, eps=1e-9):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2-x1), max(0.0, y2-y1)
    inter = iw * ih
    ua = max(0.0,(a[2]-a[0])) * max(0.0,(a[3]-a[1]))
    ub = max(0.0,(b[2]-b[0])) * max(0.0,(b[3]-b[1]))
    return inter / (ua + ub - inter + eps)

def detect_many(scene_gray, tpl_gray, sift, max_iter=20):
    h, w = tpl_gray.shape[:2]
    kp_t, des_t = sift.detectAndCompute(tpl_gray, None)
    if des_t is None or len(kp_t) < 4:
        return []

    work = scene_gray.copy()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    out = []

    for _ in range(max_iter):
        kp_s, des_s = sift.detectAndCompute(work, None)
        if des_s is None or len(kp_s) < 4:
            break

        knn = bf.knnMatch(des_t, des_s, k=2)
        good = [m for m, n in knn if m.distance < RATIO * n.distance]
        if len(good) < MIN_INLIERS:
            break

        src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THR)
        if H is None or mask.ravel().sum() < MIN_INLIERS:
            break

        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        poly = cv2.perspectiveTransform(corners, H)
        out.append((poly, bbox(poly)))

        m = np.zeros_like(work, np.uint8)
        cv2.fillPoly(m, [poly.astype(np.int32)], 255)
        work[m == 255] = 0

    return out

def nms_merge(dets, thr=IOU_MERGE):
    if not dets:
        return []
    areas = [max(0,(b[2]-b[0])) * max(0,(b[3]-b[1])) for _, b in dets]
    order = np.argsort(-np.array(areas))
    kept = []
    for i in order:
        pi, bi = dets[i]
        if all(iou(bi, bk) <= thr for _, bk in kept):
            kept.append((pi, bi))
    return kept

def put_label(img, poly, text):
    p = poly.astype(int)
    x1 = int(p[:,0,0].min()); y1 = int(p[:,0,1].min())
    # тонкая белая подложка для читабельности
    cv2.putText(img, text, (x1, max(0, y1-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x1, max(0, y1-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR, 2, cv2.LINE_AA)

def main():

    scene_path, templates_dir = sys.argv[1], sys.argv[2]

    scene_bgr  = cv2.imread(scene_path, cv2.IMREAD_COLOR)
    scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)

    entries = sorted(os.listdir(templates_dir))
    templates = []
    for fname in entries:
        fpath = os.path.join(templates_dir, fname)
        if not os.path.isfile(fpath):
            continue
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            name = os.path.splitext(fname)[0]
            if os.path.abspath(fpath) == os.path.abspath(scene_path):
                continue
            templates.append((name, img))

    sift = cv2.SIFT_create()

    for name, tpl_gray in templates:
        det1 = detect_many(scene_gray, tpl_gray, sift)
        det2 = detect_many(scene_gray, cv2.flip(tpl_gray, 1), sift)
        for poly, _ in nms_merge(det1 + det2):
            cv2.polylines(scene_bgr, [poly.astype(int)], True, COLOR, THICK)
            put_label(scene_bgr, poly, name)

    cv2.imshow("Ghosts", scene_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

#python3 halloween.py  dataset/lab7.png  dataset