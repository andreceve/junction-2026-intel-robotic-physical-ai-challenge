"""Image preprocessing shared by data collection, dataset export and the live robot runtime.

RULE: training images and live images must go through the SAME function here, otherwise the models see
something different at run time than they learned.

Background movement (people walking past the fixed cameras) is handled in two ways:
  * ACT pick policy: uses only the TOP camera cropped to the table workspace (people stand outside the crop)
    and the WRIST camera (looks down at the plate). The FRONT camera mostly sees the room -> not a policy input.
  * Anomaly images of the held brick: the front RealSense depth keeps only the nearest object cluster
    (gripper + brick) and paints everything behind it flat grey (`inhand_mask`).
"""
import cv2
import numpy as np

ACT_W, ACT_H = 640, 480     # matches the Studio environment's camera shape (480x640x3)
GREY = 128


def fit_43(box, W=1920, H=1080):
    """Grow a box (x0,y0,x1,y1) to 4:3 around its centre, kept inside the image."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w * 3 > h * 4: h = w * 3 / 4
    else: w = h * 4 / 3
    w, h = min(w, W), min(h, H)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = int(round(min(max(cx - w / 2, 0), W - w))); y0 = int(round(min(max(cy - h / 2, 0), H - h)))
    return [x0, y0, x0 + int(round(w)), y0 + int(round(h))]


def default_act_roi(cam, W=1920, H=1080):
    return fit_43([0, 0, W, H], W, H)          # centre 4:3 crop


def act_frame(img, roi):
    """Full-res BGR frame -> 640x480 BGR policy input (crop to the 4:3 `roi`, then downscale)."""
    x0, y0, x1, y1 = roi
    return cv2.resize(img[y0:y1, x0:x1], (ACT_W, ACT_H), interpolation=cv2.INTER_AREA)


def fill_holes(mask):
    """Fill everything enclosed by the outer outline (the pen cross / depth holes inside the brick stay IN)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, cv2.FILLED)
    return out


def inhand_mask(depth_mm, band_mm=70, max_mm=900, min_px=4000):
    """Mask of the nearest object cluster (gripper + held brick) from aligned depth (uint16 mm, 0 = no data).

    Finds the nearest substantial depth peak (closer than `max_mm`), keeps pixels within +-band_mm of it, takes
    the largest connected component, fills its holes and pads it slightly. Returns (mask uint8 0/255, info) or
    (None, reason) if depth is missing/unusable.
    """
    if depth_mm is None: return None, "no depth"
    d = depth_mm
    small = cv2.resize(d, (d.shape[1] // 4, d.shape[0] // 4), interpolation=cv2.INTER_NEAREST)
    v = small[(small > 150) & (small < max_mm)]
    if v.size < min_px // 16: return None, "nothing within %d mm" % max_mm
    hist, edges = np.histogram(v, bins=np.arange(150, max_mm + 20, 20))
    thr = max(min_px // 16 // 4, int(hist.max() * 0.25))
    peak = int(np.argmax(hist >= thr))                       # NEAREST bin with a substantial amount of pixels
    z = float(np.median(v[(v >= edges[peak]) & (v < edges[peak] + 40)])) if v.size else 0
    m = ((small > 0) & (np.abs(small.astype(np.int32) - z) <= band_mm)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1: return None, "no cluster"
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8) * 255
    m = fill_holes(m)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.resize(m, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST)
    return m, {"z_mm": round(z), "area_px": int(cv2.countNonZero(m))}


def apply_mask(img, mask):
    """Everything outside the mask -> flat grey (so background changes can't look like anomalies)."""
    out = np.full_like(img, GREY)
    out[mask > 0] = img[mask > 0]
    return out


def locate_brick(img, bg, roi, min_area=600, diff=40):
    """The brick lying inside `roi` (full-res x0, y0, x1, y1), seen from the overhead camera (re-grip spot).

    `bg` = the same view with the spot empty (recommended: then tape, jig walls etc. are ignored); without it
    the reference is the median colour of the roi border. Returns None if no brick, else
    {"c": (cx, cy) full-res px, "angle": long-axis angle in deg (0..sym), "long"/"short": side lengths px,
     "sym": 90 (square footprint, looks the same every 90 deg) or 180, "box": 4 corners, "edge": touches roi edge}.
    """
    x0, y0, x1, y1 = [int(v) for v in roi]
    c = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.int32)
    if bg is not None and bg.shape == img.shape:
        ref = cv2.cvtColor(bg[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.int32)
    else:
        ref = np.median(np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]]), 0)
    m = (np.abs(c - ref).sum(-1) > diff).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1: return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[k, cv2.CC_STAT_AREA] < min_area: return None
    cnts = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    pts = cv2.boxPoints(rect)
    e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
    l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    e = e1 if l1 >= l2 else e2
    lo, sh = max(l1, l2), min(l1, l2)
    sym = 90.0 if lo / max(sh, 1.0) < 1.25 else 180.0
    bx, by, bw, bh = st[k, :4]
    edge = bool(bx <= 1 or by <= 1 or bx + bw >= x1 - x0 - 1 or by + bh >= y1 - y0 - 1)
    return {"c": (float(rect[0][0] + x0), float(rect[0][1] + y0)),
            "angle": float(np.degrees(np.arctan2(e[1], e[0]))) % sym, "long": lo, "short": sh, "sym": sym,
            "box": (pts + [x0, y0]).tolist(), "edge": edge}


def ang_diff(a, b, sym):
    """a - b in degrees, wrapped into (-sym/2, sym/2] (a square brick looks the same every 90 deg)."""
    d = (a - b) % sym
    return d - sym if d > sym / 2 else d


def mask_bbox(mask, pad=0.08):
    ys, xs = np.nonzero(mask)
    if xs.size == 0: return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    p = int(max(x1 - x0, y1 - y0) * pad)
    H, W = mask.shape
    return [max(0, x0 - p), max(0, y0 - p), min(W, x1 + p), min(H, y1 + p)]
