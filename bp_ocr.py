#!/usr/bin/env python3
"""
bp_ocr.py -- read SYS / DIA / PUL from a blood-pressure monitor's LCD using a
             camera and a custom seven-segment decoder.  Medi-Kiosk / Pi 4B.

Why not ssocr: on this low-contrast reflective LCD ssocr misreads badly
(110 -> 81).  Because the display geometry is fixed, it is far more reliable to
threshold each row, find the digit blobs, and sample the seven segment regions
directly.  That is what this does, and it reads this monitor correctly.

  python3 bp_ocr.py calib     # capture + draw the ROI boxes  -> calib.png
  python3 bp_ocr.py shot      # capture one frame             -> bp.jpg
  python3 bp_ocr.py prep      # per-ROI binarised images + per-digit debug
  python3 bp_ocr.py read      # one reading
  python3 bp_ocr.py watch     # repeat until N frames agree, then print

  python3 bp_ocr.py <cmd> --file bp.jpg     # work from a saved image instead

Needs:  sudo apt install python3-opencv python3-requests
"""

import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------- config ---
URL    = "http://192.168.1.8:8080/shot.jpg"    # IP Webcam still endpoint
ROTATE = cv2.ROTATE_90_CLOCKWISE               # None once the camera is mounted upright

# Crop boxes (y1, y2, x1, x2) in pixels of the ROTATED frame.
# IMPORTANT: keep every box strictly INSIDE the grey LCD area. If a box touches
# the white bezel, Otsu latches onto bezel-vs-LCD instead of digits-vs-LCD and
# the whole row collapses into one white blob.
ROIS = {
    "sys": (585,   725, 450, 732),
    "dia": (740,   880, 450, 732),
    "pul": (920,  1055, 450, 732),
}

SEG_ON   = 0.25   # segment counts as lit when this fraction of its box is white
SCALE    = 3      # upscale factor before thresholding
MIN_AREA = 400    # ignore blobs smaller than this (specks)

LIMITS = {"sys": (60, 260), "dia": (30, 160), "pul": (30, 200)}
AGREE_FRAMES, MAX_TRIES = 3, 25

# segments ordered: a(top) f(up-left) b(up-right) g(mid) e(low-left) c(low-right) d(bottom)
SEGMAP = {
    (1,1,1,0,1,1,1): 0, (0,0,1,0,0,1,0): 1, (1,0,1,1,1,0,1): 2,
    (1,0,1,1,0,1,1): 3, (0,1,1,1,0,1,0): 4, (1,1,0,1,0,1,1): 5,
    (1,1,0,1,1,1,1): 6, (1,0,1,0,0,1,0): 7, (1,1,1,1,1,1,1): 8,
    (1,1,1,1,0,1,1): 9,
}
# ---------------------------------------------------------------------------


def capture():
    """Grab one frame from the network camera."""
    import requests
    r = requests.get(URL, timeout=5)
    r.raise_for_status()
    frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("could not decode image from camera")
    return cv2.rotate(frame, ROTATE) if ROTATE is not None else frame


def get_frame(path=None):
    if path:
        img = cv2.imread(path)
        if img is None:
            raise SystemExit(f"cannot read {path}")
        return img
    return capture()


def binarize(frame, box):
    """Crop one digit row -> white digits on black."""
    y1, y2, x1, x2 = box
    roi = cv2.resize(frame[y1:y2, x1:x2], None, fx=SCALE, fy=SCALE,
                     interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (7, 7), 0)
    # Digits are darker than the LCD background -> INV makes them white.
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Seven-segment strokes are chunky; OPEN clears speckle without eating them.
    return cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))


def digit_boxes(th, min_h_frac=0.45):
    """Group blobs into digits. The segments of one digit don't touch, so we
    merge blobs that overlap horizontally."""
    H, W = th.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats((th > 128).astype(np.uint8), 8)
    parts = []
    for i in range(1, n):
        x, y, w, h, a = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]
        if a < MIN_AREA:
            continue
        edge = y <= 1 or y + h >= H - 1 or x <= 1 or x + w >= W - 1
        if edge and h < 0.30 * H and w < 0.30 * W:
            continue                      # noise clinging to the crop border
        parts.append((x, y, w, h))
    if not parts:
        return []

    parts.sort(key=lambda p: p[0])
    groups = [[parts[0]]]
    for p in parts[1:]:
        if p[0] < max(q[0] + q[2] for q in groups[-1]):
            groups[-1].append(p)
        else:
            groups.append([p])

    boxes = []
    for g in groups:
        x1 = min(q[0] for q in g); x2 = max(q[0] + q[2] for q in g)
        y1 = min(q[1] for q in g); y2 = max(q[1] + q[3] for q in g)
        if y2 - y1 >= min_h_frac * H:
            boxes.append((x1, y1, x2, y2))
    return boxes


def decode_digit(th, b, debug=False):
    """Sample the seven segment zones of one digit and look the pattern up."""
    x1, y1, x2, y2 = b
    d = th[y1:y2, x1:x2]
    h, w = d.shape
    if w < 0.28 * h:                       # a '1' is just two thin strokes
        return (1, None) if debug else 1

    def frac(ya, yb, xa, xb):
        r = d[int(ya * h):int(yb * h), int(xa * w):int(xb * w)]
        return float((r > 128).mean()) if r.size else 0.0

    f = (frac(0.00, 0.18, 0.25, 0.75),     # a
         frac(0.10, 0.45, 0.00, 0.22),     # f
         frac(0.10, 0.45, 0.78, 1.00),     # b
         frac(0.42, 0.58, 0.25, 0.75),     # g
         frac(0.55, 0.90, 0.00, 0.22),     # e
         frac(0.55, 0.90, 0.78, 1.00),     # c
         frac(0.82, 1.00, 0.25, 0.75))     # d
    key = tuple(1 if s > SEG_ON else 0 for s in f)
    val = SEGMAP.get(key)
    return (val, [round(s, 2) for s in f]) if debug else val


def read_row(frame, box, debug=False):
    th = binarize(frame, box)
    out = []
    for b in digit_boxes(th):
        v = decode_digit(th, b, debug=debug)
        if debug:
            v, f = v
            print(f"      digit {b} -> {v}   segs {f}")
        out.append(str(v) if v is not None else "?")
    s = "".join(out)
    return int(s) if s.isdigit() and s else None


def read_once(frame=None, debug=False):
    frame = frame if frame is not None else capture()
    res = {}
    for name, box in ROIS.items():
        if debug:
            print(f"   {name}:")
        v = read_row(frame, box, debug=debug)
        lo, hi = LIMITS[name]
        res[name] = v if (v is not None and lo <= v <= hi) else None
    return res


# ------------------------------------------------------------- commands ----
def cmd_calib(frame):
    ov = frame.copy()
    col = {"sys": (0, 0, 255), "dia": (0, 180, 0), "pul": (255, 0, 0)}
    for name, (y1, y2, x1, x2) in ROIS.items():
        c = col[name]
        cv2.rectangle(ov, (x1, y1), (x2, y2), c, 4)
        cv2.putText(ov, name.upper(), (x1 - 95, y1 + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, c, 3)
    cv2.imwrite("calib.png", ov)
    print(f"frame {frame.shape} -> calib.png")
    print("each box must sit fully inside the grey LCD and hug one row of digits")


def cmd_shot(frame):
    cv2.imwrite("bp.jpg", frame)
    print(f"saved bp.jpg {frame.shape}")


def cmd_prep(frame):
    for name, box in ROIS.items():
        cv2.imwrite(f"prep_{name}.png", binarize(frame, box))
        print(f"   {name}: prep_{name}.png")
    print()
    r = read_once(frame, debug=True)
    print(f"\nSYS:{r['sys']} DIA:{r['dia']} PUL:{r['pul']}")


def cmd_read(frame):
    r = read_once(frame)
    print(f"SYS:{r['sys']} DIA:{r['dia']} PUL:{r['pul']}")


def cmd_watch(_frame):
    last, streak = None, 0
    for i in range(MAX_TRIES):
        r = read_once()
        key = (r["sys"], r["dia"], r["pul"])
        ok = all(v is not None for v in key)
        streak = streak + 1 if (ok and key == last) else (1 if ok else 0)
        last = key if ok else None
        print(f"  [{i+1:2d}] {key}  streak={streak}")
        if streak >= AGREE_FRAMES:
            print(f"\nSYS:{key[0]} DIA:{key[1]} PUL:{key[2]}")
            return
        time.sleep(0.4)
    print("\nno stable reading -- check framing, focus, glare")


if __name__ == "__main__":
    args = sys.argv[1:]
    path = None
    if "--file" in args:
        i = args.index("--file")
        path = args[i + 1]
        args = args[:i] + args[i + 2:]
    cmd = args[0] if args else ""
    cmds = {"calib": cmd_calib, "shot": cmd_shot, "prep": cmd_prep,
            "read": cmd_read, "watch": cmd_watch}
    if cmd not in cmds:
        print(__doc__)
        sys.exit(0)
    frame = None if cmd == "watch" else get_frame(path)
    cmds[cmd](frame)