"""Stage 4: map players onto a top-down rink.

The ice is a plane, so image ↔ rink is a homography. On the first frame it comes from 4 known points
(the neutral-zone faceoff dots). The camera pans and zooms, so for every next frame we estimate the
frame-to-previous-frame homography from ORB matches on the ice (players masked out) with RANSAC, and chain:
    H_rink<-k = H_rink<-0 · H_0<-1 · ... · H_k-1<-k
"""
import json
import pickle
from collections import Counter

import cv2
import numpy as np

from hockey.render import COLORS, draw
from hockey.track import STEP, ice_region, read_frames

# NHL rink in feet, origin at centre ice, x along the length, y across (far side positive)
L, W = 200, 85
KEY_IMG = np.float32([[296, 241], [985, 247], [104, 536], [1107, 551]])    # faceoff dots on frame 0 (1280 px wide)
KEY_RINK = np.float32([[-20, 22], [20, 22], [-20, -22], [20, -22]])


def feature_mask(frame, dets):
    m = ice_region(frame) * 255
    # the broadcast overlays (score bug, channel logo) never move and would pin the homography to identity
    m[:int(0.2 * m.shape[0])] = 0
    for _, x1, y1, x2, y2, _ in dets:
        cv2.rectangle(m, (int(x1) - 5, int(y1) - 5), (int(x2) + 5, int(y2) + 15), 0, -1)
    return m


MIN_INLIERS = 25        # frame-to-frame: consecutive frames, so even a blurred pan gives a usable fit
RELOC_INLIERS = 80      # relocalization against an older keyframe needs much stronger evidence


def match_homography(bf, kp, des, kp2, des2):
    """Homography mapping points of the first frame to the second, and its RANSAC inlier count."""
    if des is None or des2 is None:
        return None, 0
    m = bf.match(des, des2)
    if len(m) < 12:
        return None, 0
    a = np.float32([kp[x.queryIdx].pt for x in m]); b = np.float32([kp2[x.trainIdx].pt for x in m])
    G, mask = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
    return (G, int(mask.sum())) if G is not None else (None, 0)


def colour_hist(frame):
    h = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def frame_homographies(src, tracks):
    """Returns H[k] mapping frame k pixels to rink feet (None while the camera is lost), and per-frame status.

    Within a shot, homographies are chained frame to frame. A cut (sudden colour-histogram change or too
    few RANSAC inliers) breaks the chain; from then on every frame is matched against a bank of earlier
    keyframes with known H until one matches well (relocalization). Chaining drifts, so every 10 frames the
    current frame is also matched against keyframes from the opening seconds (still drift-free); a strong
    match replaces the chained estimate.
    """
    orb = cv2.ORB_create(3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    H0 = cv2.getPerspectiveTransform(KEY_IMG, KEY_RINK)
    Hs, status, prev, bank, anchors, last_hist = [], [], None, [], [], None
    for k, frame in enumerate(read_frames(src)):
        if k >= len(tracks):
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, feature_mask(frame, tracks[k]))
        hist = colour_hist(frame)
        cut = last_hist is not None and cv2.compareHist(hist, last_hist, cv2.HISTCMP_CORREL) < 0.8
        last_hist = hist
        H, st = None, "lost"
        if k == 0:
            H, st = H0, "anchor"
        elif Hs[-1] is not None and not cut:
            G, n = match_homography(bf, kp, des, *prev)
            if n >= MIN_INLIERS:
                H, st = Hs[-1] @ G, "tracked"
        if H is not None and k % 10 == 0 and anchors:    # drift correction against the accurate opening keyframes
            (G, n), aH = max(((match_homography(bf, kp, des, akp, ades), aH) for akp, ades, aH in anchors),
                             key=lambda r: r[0][1])
            if n >= RELOC_INLIERS:
                H, st = aH @ G, "corrected"
        if H is None and bank and k % 2 == 0:            # relocalize (every other frame, to save time)
            best = max(((match_homography(bf, kp, des, bkp, bdes), bH) for bkp, bdes, bH in bank[::max(1, len(bank) // 40)]),
                       key=lambda r: r[0][1])
            (G, n), bH = best
            if n >= RELOC_INLIERS:
                H, st = bH @ G, "relocalized"
        if H is not None and k % 15 == 0 and st != "relocalized":
            bank.append((kp, des, H))
            if k < 150:                                   # before the first long pan: no drift yet
                anchors.append((kp, des, H))
        Hs.append(H); status.append(st)
        prev = (kp, des)
    return Hs, status


def rink_canvas(scale=5, pad=20):
    w, h = int(L * scale + 2 * pad), int(W * scale + 2 * pad)
    img = np.full((h, w, 3), 245, np.uint8)
    to_px = lambda x, y: (int(pad + (x + L / 2) * scale), int(pad + (W / 2 - y) * scale))
    cv2.rectangle(img, to_px(-L / 2, W / 2), to_px(L / 2, -W / 2), (60, 60, 60), 2)
    cv2.line(img, to_px(0, W / 2), to_px(0, -W / 2), (40, 40, 210), 3)                       # centre red line
    for x in (-25, 25):
        cv2.line(img, to_px(x, W / 2), to_px(x, -W / 2), (200, 90, 30), 3)                  # blue lines
    for x in (-89, 89):
        cv2.line(img, to_px(x, W / 2 - 4), to_px(x, -W / 2 + 4), (40, 40, 210), 1)         # goal lines
    cv2.circle(img, to_px(0, 0), 15 * scale, (200, 90, 30), 1)
    for x, y in [(-69, 22), (-69, -22), (69, 22), (69, -22)]:
        cv2.circle(img, to_px(x, y), 15 * scale, (40, 40, 210), 1)
    for x, y in [(-20, 22), (20, 22), (-20, -22), (20, -22), (-69, 22), (-69, -22), (69, 22), (69, -22)]:
        cv2.circle(img, to_px(x, y), 4, (40, 40, 210), -1)
    return img, to_px


def run(src="data/clip.mov", tracks_path="runs/tracks.pkl", teams_path="runs/teams.json", out="runs/minimap.mp4"):
    tracks = pickle.load(open(tracks_path, "rb")); team = json.load(open(teams_path))
    Hs, status = frame_homographies(src, tracks)
    print("camera status per frame:", dict(Counter(status)))
    print("relocalized at frames:", [k for k, s in enumerate(status) if s == "relocalized"][:20])
    base, to_px = rink_canvas()
    fps = cv2.VideoCapture(src).get(cv2.CAP_PROP_FPS) / STEP
    writer, positions = None, []
    for k, frame in enumerate(read_frames(src)):
        if k >= len(Hs):
            break
        img = draw(frame, tracks[k], team)
        mini = base.copy()
        h, w = frame.shape[:2]
        cur = []
        if Hs[k] is None:
            cv2.putText(mini, "camera not located (close-up / cut)", (40, mini.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 3)
        else:
            draw_on_rink(mini, Hs[k], tracks[k], team, to_px, (h, w), cur)
        positions.append(cur)
        mh = int(mini.shape[0] * 0.42 * w / mini.shape[1])
        small = cv2.resize(mini, (int(w * 0.42), mh))
        img[h - mh - 10:h - 10, w - small.shape[1] - 10:w - 10] = small
        if writer is None:
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(img)
    writer.release()
    pickle.dump((positions, status), open("runs/positions.pkl", "wb"))
    print("saved", out)


def draw_on_rink(mini, H, dets, team, to_px, shape, cur):
    h, w = shape
    # camera footprint: where the image corners land on the rink
    corners = cv2.perspectiveTransform(np.float32([[[0, h * 0.25], [w, h * 0.25], [w, h], [0, h]]]), H)[0]
    cv2.polylines(mini, [np.int32([to_px(*p) for p in corners])], True, (170, 170, 170), 1)
    for tid, x1, y1, x2, y2, conf in dets:
        name = team.get(str(tid))
        if name is None:
            continue
        fx, fy = cv2.perspectiveTransform(np.float32([[[(x1 + x2) / 2, y2]]]), H)[0, 0]
        cur.append((tid, name, float(fx), float(fy)))
        if abs(fx) <= L / 2 + 5 and abs(fy) <= W / 2 + 5:
            cv2.circle(mini, to_px(fx, fy), 7, COLORS[name], -1)


if __name__ == "__main__":
    run()
