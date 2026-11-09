"""Stage 2: assign every track to EDM, WPG or referee from the colours of the jersey.

For the middle of the upper body (the jersey, away from the ice around the player) we measure:
  dark  - share of dark pixels (V < 90)
  white - share of white pixels (low saturation, V > 165)
  sat   - mean saturation of the dark pixels: navy is saturated, the referees' black stripes are not
  blue  - mean (B - R) of the dark pixels: navy is bluish, black is neutral
Every frame of a track is classified with simple rules, and the track takes the majority label:
  referee: some dark, and the dark is unsaturated and not blue (black stripes)
  WPG:     mostly dark (navy home jersey)
  EDM:     mostly white (white away jersey), with the white share breaking ties in between

A first version clustered ImageNet ResNet-18 embeddings of the crops with k-means. It grouped players by
pose, motion blur and background rather than by jersey, so the colour rules replaced it.
"""
import json
import pickle
from collections import Counter, defaultdict

import cv2
import numpy as np

from hockey.track import read_frames


def jersey_features(frame, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    bh, bw = y2 - y1, x2 - x1
    crop = frame[y1 + int(0.2 * bh):y1 + int(0.5 * bh), x1 + int(0.3 * bw):x2 - int(0.3 * bw)]
    if crop.size < 30:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    bgr = crop.reshape(-1, 3).astype(float)
    dark = hsv[:, 2] < 90
    white = (hsv[:, 1] < 45) & (hsv[:, 2] > 165)
    if dark.sum() > 5:
        sat, blue = hsv[dark, 1].mean(), (bgr[dark, 0] - bgr[dark, 2]).mean()
    else:
        sat, blue = 0.0, 0.0
    return dark.mean(), white.mean(), sat, blue


def classify(dark, white, sat, blue):
    if dark >= 0.2 and sat < 66 and blue < 5:
        return "REF"
    if dark >= 0.40:
        return "WPG"
    if dark < 0.30:
        return "EDM"
    return "EDM" if white >= 0.2 else "WPG"


def run(video="data/clip.mov", tracks_path="runs/tracks.pkl", out="runs/teams.json"):
    tracks = pickle.load(open(tracks_path, "rb"))
    feats = defaultdict(list)
    for k, frame in enumerate(read_frames(video)):
        if k >= len(tracks):
            break
        for tid, x1, y1, x2, y2, conf in tracks[k]:
            if conf < 0.4 or (y2 - y1) < 40:
                continue
            f = jersey_features(frame, (x1, y1, x2, y2))
            if f is not None:
                feats[tid].append(f)
    # majority vote over the track's frames: robust to frames where another player overlaps the box
    team = {int(t): Counter(classify(*f) for f in v).most_common(1)[0][0] for t, v in feats.items()}
    json.dump(team, open(out, "w"))
    print("tracks per group:", dict(Counter(team.values())))
    print("saved", out)
    return team


if __name__ == "__main__":
    run()
