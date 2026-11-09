"""Stage 3: draw tracks coloured by team, and a contact sheet of crops for checking the team split."""
import json
import pickle
from collections import defaultdict

import cv2
import numpy as np

from hockey.track import STEP, read_frames

COLORS = {"EDM": (0, 110, 255), "WPG": (160, 50, 10), "REF": (30, 30, 30)}   # BGR


def draw(frame, dets, team):
    img = frame.copy()
    for tid, x1, y1, x2, y2, conf in dets:
        name = team.get(str(tid))
        if name is None:
            continue
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, COLORS[name], 2)
        cv2.putText(img, f"{name} {tid}", (p1[0], p1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[name], 2)
    return img


def video(src="data/clip.mov", tracks_path="runs/tracks.pkl", teams_path="runs/teams.json", out="runs/tracked.mp4", fps=None):
    tracks = pickle.load(open(tracks_path, "rb")); team = json.load(open(teams_path))
    fps = fps or cv2.VideoCapture(src).get(cv2.CAP_PROP_FPS) / STEP
    writer = None
    for k, frame in enumerate(read_frames(src)):
        if k >= len(tracks):
            break
        if writer is None:
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        writer.write(draw(frame, tracks[k], team))
    writer.release()
    print("saved", out)


def contact_sheet(src="data/clip.mov", tracks_path="runs/tracks.pkl", teams_path="runs/teams.json", out="runs/teams_sheet.jpg", per_team=24):
    tracks = pickle.load(open(tracks_path, "rb")); team = json.load(open(teams_path))
    best = {}                                     # track id -> (conf, frame index, box), one crop per track
    for k, dets in enumerate(tracks):
        for tid, x1, y1, x2, y2, conf in dets:
            if str(tid) in team and (y2 - y1) > 60 and conf > best.get(tid, (0,))[0]:
                best[tid] = (conf, k, (x1, y1, x2, y2))
    want = defaultdict(list)
    for tid, (c, k, box) in best.items():
        want[k].append((tid, box))
    crops = defaultdict(list)
    for k, frame in enumerate(read_frames(src)):
        for tid, (x1, y1, x2, y2) in want.get(k, []):
            crop = cv2.resize(frame[int(y1):int(y2), int(x1):int(x2)], (60, 110))
            crops[team[str(tid)]].append(crop)
    rows = []
    for name in ("EDM", "WPG", "REF"):
        cs = crops[name][:per_team] + [np.full((110, 60, 3), 255, np.uint8)] * max(0, per_team - len(crops[name]))
        label = np.full((110, 80, 3), 255, np.uint8)
        cv2.putText(label, name, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[name], 2)
        rows.append(np.hstack([label] + cs))
    cv2.imwrite(out, np.vstack(rows))
    print("saved", out, {n: len(c) for n, c in crops.items()})


if __name__ == "__main__":
    contact_sheet()
    video()
