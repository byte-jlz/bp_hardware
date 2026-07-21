#!/usr/bin/env python3
"""
bp_ocr.py -- read SYS / DIA / PUL off a blood-pressure monitor's LCD with a
             camera + seven-segment OCR, for the Medi-Kiosk Pi 4B.

  python3 bp_ocr.py calib          # capture + draw ROI boxes -> calib.png
  python3 bp_ocr.py shot           # capture one frame -> bp.jpg
  python3 bp_ocr.py prep           # write thresholded ROI images to inspect
  python3 bp_ocr.py read           # one reading
  python3 bp_ocr.py watch          # keep reading until N frames agree

Needs:  sudo apt install python3-opencv python3-requests ssocr
"""

import subprocess
import sys
import time

import cv2
import numpy as np
import requests

# ---------------------------------------------------------------- config ---
URL      = "http://192.168.1.8:8080/shot.jpg"   # IP Webcam still endpoint
ROTATE   = cv2.ROTATE_90_CLOCKWISE              # set to None once mounted upright

# Crop boxes as (y1, y2, x1, x2), in pixels of the ROTATED frame.
# Run `calib` and adjust these until each box tightly hugs one row of digits.
ROIS = {
    "sys": (730,  870, 470, 710),
    "dia": (870, 1020, 470, 710),
    "pul": (1030, 1170, 470, 710),
}

# Plausibility limits -- a reading outside these is treated as a misread.
LIMITS = {"sys": (60, 260), "dia": (30, 160), "pul": (30, 200)}

AGREE_FRAMES = 3      # how many consecutive identical readings to trust
MAX_TRIES    = 25     # give up after this many frames
# ---------------------------------------------------------------------------


def capture():
    """Grab one frame from the camera and return it as a BGR image."""
    r = requests.get(URL, timeout=5)
    r.raise_for_status()
    frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("could not decode image from camera")
    if ROTATE is not None:
        frame = cv2.rotate(frame, ROTATE)
    return frame


def prep_roi(frame, box, scale=3):
    """Crop one digit row and turn it into clean white-on-black for ssocr."""
    y1, y2, x1, x2 = box
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # The LCD is low-contrast grey-on-grey, so lift local contrast first.
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    # Digits are DARKER than the background -> INV so they come out white.
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 71, 12)
    # Close small gaps inside segments, then drop specks.
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    return th


def ocr(th):
    """Run ssocr on a prepared image. Returns an int, or None if unreadable."""
    cv2.imwrite("/tmp/_ssocr.png", th)
    try:
        out = subprocess.run(
            ["ssocr", "-d", "-1", "--number-pixels=3", "invert", "/tmp/_ssocr.png"],
            capture_output=True, text=True, timeout=10)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    digits = "".join(c for c in out.stdout if c.isdigit())
    return int(digits) if digits else None


def read_once(frame=None):
    """Read all three fields from one frame. Values may be None."""
    if frame is None:
        frame = capture()
    result = {}
    for name, box in ROIS.items():
        val = ocr(prep_roi(frame, box))
        lo, hi = LIMITS[name]
        result[name] = val if (val is not None and lo <= val <= hi) else None
    return result


# ------------------------------------------------------------- commands ----
def cmd_calib():
    frame = capture()
    ov = frame.copy()
    colours = {"sys": (0, 0, 255), "dia": (0, 180, 0), "pul": (255, 0, 0)}
    for name, (y1, y2, x1, x2) in ROIS.items():
        c = colours[name]
        cv2.rectangle(ov, (x1, y1), (x2, y2), c, 4)
        cv2.putText(ov, name.upper(), (x1 - 95, y1 + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, c, 3)
    cv2.imwrite("calib.png", ov)
    print(f"frame {frame.shape} -> calib.png  (check each box hugs one row)")


def cmd_shot():
    frame = capture()
    cv2.imwrite("bp.jpg", frame)
    print(f"saved bp.jpg {frame.shape}")


def cmd_prep():
    frame = capture()
    for name, box in ROIS.items():
        th = prep_roi(frame, box)
        cv2.imwrite(f"prep_{name}.png", th)
        print(f"prep_{name}.png  ->  ssocr says: {ocr(th)}")
    print("\nWant: solid white digits, black background, nothing else.")


def cmd_read():
    r = read_once()
    print(f"SYS:{r['sys']} DIA:{r['dia']} PUL:{r['pul']}")


def cmd_watch():
    """Only accept a reading once it repeats identically N times."""
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
    cmds = {"calib": cmd_calib, "shot": cmd_shot, "prep": cmd_prep,
            "read": cmd_read, "watch": cmd_watch}
    fn = cmds.get(sys.argv[1] if len(sys.argv) > 1 else "")
    if fn:
        fn()
    else:
        print(__doc__)