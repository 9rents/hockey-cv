"""Stage 1: detect and track people on the ice. Saves boxes per frame to runs/tracks.pkl."""
import pickle
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

WIDTH = 1280            # frames are resized to this width
STEP = 2                # use every 2nd frame (~28 fps)


def read_frames(video, max_frames=None):
    cap = cv2.VideoCapture(str(video))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % STEP == 0:
            yield cv2.resize(frame, (WIDTH, int(frame.shape[0] * WIDTH / frame.shape[1])))
            if max_frames and i // STEP + 1 >= max_frames:
                break
        i += 1


def ice_region(frame):
    """The ice surface: convex hull of the large bright, unsaturated regions.
    The coloured kick plate at the bottom of the boards separates the ice from the white boards above it,
    and the boards strip is thin, so it is dropped by the height test. The hull bridges the painted
    lines, logos and players that cut the white area into pieces."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 50) & (hsv[..., 2] > 160)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    H, W = white.shape
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0.02 * H * W and stats[i, cv2.CC_STAT_HEIGHT] > 0.2 * H]
    if not keep:
        keep = [1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])]
    pts = np.vstack([cv2.findNonZero((lab == i).astype(np.uint8)) for i in keep])
    ice = np.zeros_like(white)
    cv2.fillConvexPoly(ice, cv2.convexHull(pts), 1)
    return ice


def on_ice(box, ice):
    x1, y1, x2, y2 = box.astype(int)
    h, w = ice.shape
    cx = (x1 + x2) // 2
    y = min(y2 - 2, h - 1)
    patch = ice[max(y - 4, 0):y + 4, max(cx - 6, 0):cx + 6]
    return patch.size > 0 and patch.mean() > 0.5


def run(video, out="runs/tracks.pkl", max_frames=None):
    model = YOLO("yolo11m.pt")
    tracks = []                                  # per frame: list of (track_id, x1, y1, x2, y2, conf)
    for k, frame in enumerate(read_frames(video, max_frames)):
        r = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=0.25,
                        imgsz=1280, device="mps", verbose=False)[0]
        ice = ice_region(frame)
        cur = []
        if r.boxes.id is not None:
            for box, tid, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist(), r.boxes.conf.cpu().tolist()):
                if on_ice(box, ice):
                    cur.append((tid, *box.tolist(), c))
        tracks.append(cur)
        if k % 200 == 0:
            print(f"frame {k}: {len(cur)} people on ice")
    Path(out).parent.mkdir(exist_ok=True)
    pickle.dump(tracks, open(out, "wb"))
    print("saved", out, len(tracks), "frames")


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "data/clip.mov", max_frames=int(sys.argv[2]) if len(sys.argv) > 2 else None)
