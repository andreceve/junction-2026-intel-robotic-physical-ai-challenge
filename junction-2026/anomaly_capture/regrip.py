"""Re-grip station: see the faces that were under the gripper fingers (phase 5, "check every face").

The gripper always covers 2 faces of the brick. While it is held, the front camera already sees the other side
faces and the bottom (the WP2 photo poses); the top face is seen by the top camera before the pick. To see the
covered faces the robot sets the brick down on a FIXED spot (the "re-grip spot", ideally an L-shaped corner /
jig taped to the table), lets go, looks at it with the TOP camera and grips it again:

  turn   (square 2x2 footprint): gripped again turned 90 deg -> the 2 covered faces are now free.
  shift  (long 2x4 footprint, too long to clamp across): gripped again further along its length, on the
         other side of where the fingers were -> the strip of the long faces under the fingers is now free.

Which one is used is decided from the brick's footprint measured on the spot (`plans` in the settings).

Why the top camera: when the brick is let go it can slide/turn a little, and the SO-101 gripper has one FIXED
jaw, so a held brick is never centred on the wrist-roll axis: a blind "turn the wrist 90 deg and close" lands
the fingers on the brick's edge. So after letting go the arm moves out of the top camera's view (`regrip_clear`),
the brick's centre + angle are measured and the grip pose is corrected:
  * angle -> wrist roll (exactly 90 deg relative to the brick, not to the wrist);
  * position -> a local linear model, NO inverse kinematics needed: besides the spot pose (`tape_place`) two
    more poses are taught 1-2 cm away in two directions (`tape_dx`, `tape_dy`). The calibration measures where a
    held brick lands for each of them; any correction is then a blend of the three taught poses.

Calibration (button, ~1 min, with a SQUARE brick held centred between the fingers): put/measure/grip again at
centre, centre + 20 deg roll, dx, dy and centre turned 90 deg. Stored in collect_settings.json ("regrip.cal");
re-calibrate whenever the spot poses are re-taught (detected automatically).

Without the clear pose / spot area / calibration it falls back to the blind turn (old behaviour).
"""
import time
from datetime import datetime

import cv2
import numpy as np

import prep
from arm import BODY, TICK_PER_DEG

DEFAULTS = {"roi": None, "open_frac": 0.75, "max_tries": 2, "shift_mm": 18.0, "short_mm": 31.8,
            "plans": {"square": ["turn"], "long": ["shift"]}, "cal": None}
SLOTS = ("tape_above", "tape_place", "tape_dx", "tape_dy", "regrip_clear")
MAX_BLEND = 2.5          # correction may extrapolate at most 2.5x the taught dx/dy offsets


class Regrip:
    def __init__(self, C):
        self.C = C
        cfg = C.cfg.setdefault("regrip", {})
        for k, v in DEFAULTS.items(): cfg.setdefault(k, v)
        self.bg_path = C.DATA / "top" / "regrip_bg.png"
        self.bg = cv2.imread(str(self.bg_path)) if self.bg_path.exists() else None
        self.vis = None           # last annotated measurement (jpg bytes) for the page
        self.last = None          # last measurement + decision (shown on the page)
        self.shape = None         # footprint of the brick being inspected ("square" / "long")

    # ------------------------------------------------------------------ setup state
    @property
    def cfg(self): return self.C.cfg["regrip"]

    def pose(self, name):
        w = self.C.slot(name)
        return {k: float(v) for k, v in w[0]["pos"].items()} if w else None

    def spot_taught(self): return bool(self.pose("tape_above") and self.pose("tape_place"))

    def cal_stale(self):
        cal, p = self.cfg.get("cal"), self.pose("tape_place")
        return bool(cal and p and any(abs(cal["place"][n] - p[n]) > 3 for n in BODY))

    def missing(self):
        """What is still missing for the camera-corrected re-grip (empty list = ready)."""
        out = [f"teach '{s}'" for s in SLOTS if not self.pose(s)]
        if not self.cfg.get("roi"): out.append("drag the re-grip spot area on the top view")
        if "top" not in self.C.S.live: out.append("top camera offline")
        if not self.cfg.get("cal"): out.append("CALIBRATE")
        elif self.cal_stale(): out.append("spot pose re-taught: CALIBRATE again")
        return out

    def plan(self):
        return self.cfg["plans"].get(self.shape or "square", ["turn"])

    def summary(self):
        cal = self.cfg.get("cal")
        return {"roi": self.cfg.get("roi"), "open_frac": self.cfg["open_frac"], "shift_mm": self.cfg["shift_mm"],
                "max_tries": self.cfg["max_tries"], "plans": self.cfg["plans"], "has_bg": self.bg is not None,
                "calibrated": bool(cal), "cal_time": cal and cal.get("time"), "stale": self.cal_stale(),
                "missing": self.missing(), "last": self.last}

    # ------------------------------------------------------------------ vision
    def save_bg(self):
        shots = self.C.S.averaged(["top"])
        if "top" not in shots: raise RuntimeError("top camera gave no frames")
        self.bg_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.bg_path), shots["top"][0]); self.bg = shots["top"][0]

    def measure(self):
        """Averaged top frame -> brick on the spot (prep.locate_brick) or None. Also renders self.vis."""
        roi = self.cfg.get("roi")
        if not roi: return None
        shots = self.C.S.averaged(["top"])
        if "top" not in shots: return None
        f = shots["top"][0]
        m = prep.locate_brick(f, self.bg, roi)
        self.render(f, m)
        return m

    def render(self, f, m, want=None, ref=None):
        x0, y0, x1, y1 = self.cfg["roi"]; p = 60
        X0, Y0 = max(0, x0 - p), max(0, y0 - p)
        v = f[Y0:min(f.shape[0], y1 + p), X0:min(f.shape[1], x1 + p)].copy()
        off = np.array([X0, Y0])
        cv2.rectangle(v, (x0 - X0, y0 - Y0), (x1 - X0, y1 - Y0), (255, 200, 0), 2)
        if m:
            cv2.polylines(v, [(np.array(m["box"]) - off).astype(np.int32)], True, (0, 255, 0), 2)
            cv2.circle(v, tuple((np.array(m["c"]) - off).astype(int)), 4, (0, 255, 0), -1)
        if ref is not None: cv2.drawMarker(v, tuple((np.array(ref) - off).astype(int)), (255, 120, 0), cv2.MARKER_CROSS, 18, 2)
        if want is not None: cv2.drawMarker(v, tuple((np.array(want) - off).astype(int)), (0, 220, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
        sc = 360 / max(1, v.shape[0])
        v = cv2.resize(v, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
        self.vis = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    # ------------------------------------------------------------------ motion primitives
    def above_of(self, pose):
        """Straight above `pose`: the taught 'above the spot' shifted by the same joint offsets as `pose`."""
        P0, U = self.pose("tape_place"), self.pose("tape_above")
        return {**{n: U[n] + (pose[n] - P0[n]) for n in BODY}, "gripper": pose.get("gripper", U["gripper"])}

    def g_wide(self, g_open):
        gmin, gmax = self.C.A.fl["gripper"]
        return max(g_open, gmin + self.cfg["open_frac"] * (gmax - gmin))

    def roll_dir(self):
        lo, hi = self.C.A.fl["wrist_roll"]
        return 1 if self.pose("tape_place")["wrist_roll"] + 90 * TICK_PER_DEG <= hi - 20 else -1

    def fit_roll(self, r, sym):
        """Same grip, different roll: r +- n*sym (the brick looks the same) -> the one inside the joint limits."""
        lo, hi = self.C.A.fl["wrist_roll"]
        c = [r + n * sym * TICK_PER_DEG for n in (0, -1, 1, -2, 2)]
        ok = [x for x in c if lo + 20 <= x <= hi - 20]
        if not ok: raise RuntimeError("re-grip: wrist roll needed is outside the joint limits")
        return ok[0]

    def put(self, pose, g_lock, g_open):
        """Held brick: above -> slowly down to `pose` -> open slowly -> straight up (gripper open)."""
        C, spd = self.C, self.C.cfg["speed"]
        up = self.above_of(pose)
        C.goto(up, gripper=g_lock); C.check()
        C.goto(pose, spd * 0.3, gripper=g_lock)
        C.goto(dict(C.A.cmd), 0.3, gripper=g_open); time.sleep(0.2)
        C.goto(up, spd * 0.5, gripper=g_open)

    def look(self, g_open):
        """Arm out of the top camera's view, then measure."""
        clear = self.pose("regrip_clear")
        if clear: self.C.goto(clear, gripper=g_open)
        self.C.check(); time.sleep(0.3)
        return self.measure()

    def take(self, pose, g_lock, g_wide):
        """Gripper wide open: above -> slowly down -> close with the SAME clamp command -> straight up."""
        C, spd = self.C, self.C.cfg["speed"]
        up = self.above_of(pose)
        C.goto(up, gripper=g_wide); C.check()
        C.goto(pose, spd * 0.3, gripper=g_wide)
        C.goto(dict(C.A.cmd), 0.5, gripper=g_lock); time.sleep(0.25)
        held = C.holding()
        C.goto(up, spd * 0.5, gripper=g_lock)
        return held and C.holding()

    # ------------------------------------------------------------------ calibration
    def calibrate(self, g_lock, g_open):
        """Holding a SQUARE brick centred between the fingers. ~1 min. Ends holding it above the spot."""
        miss = [m for m in self.missing() if "CALIBRATE" not in m]
        if miss: raise RuntimeError("before calibrating: " + "; ".join(miss))
        if self.bg is None: self.C.say("no empty-spot reference saved: using the spot's border colour (less robust)", False)
        P0, s, gw = self.pose("tape_place"), self.roll_dir(), self.g_wide(g_open)
        rot = lambda p, d: {**p, "wrist_roll": p["wrist_roll"] + d * TICK_PER_DEG}
        steps = [("centre", P0), ("roll+20", rot(P0, 20 * s)), ("dx", self.pose("tape_dx")),
                 ("dy", self.pose("tape_dy")), ("turned", rot(P0, 90 * s))]
        meas = {}
        for i, (name, p) in enumerate(steps, 1):
            self.C.say(f"calibrating re-grip {i}/{len(steps)}: {name}: set down, look, grip again")
            self.put(p, g_lock, g_open)
            m = self.look(g_open)
            if m is None: raise RuntimeError(f"calibration '{name}': no brick seen on the spot (check the spot area / reference)")
            meas[name] = m
            if not self.take(p, g_lock, gw): raise RuntimeError(f"calibration '{name}': brick not gripped again")
        m0 = meas["centre"]
        if m0["sym"] != 90: raise RuntimeError("calibrate with a SQUARE (2x2) brick")
        k = prep.ang_diff(meas["roll+20"]["angle"], m0["angle"], 90) / (20 * s)
        if not 0.5 <= abs(k) <= 1.5: raise RuntimeError(f"calibration: 20 deg of wrist roll turned the brick {k * 20:.0f} deg in the image (expected ~±20): the brick slips?")
        k = float(np.sign(k))     # an overhead camera sees a roll about the vertical 1:1; only the sign (mirror) is unknown
        c0 = np.array(m0["c"])
        J = np.column_stack([np.array(meas["dx"]["c"]) - c0, np.array(meas["dy"]["c"]) - c0])
        if abs(np.linalg.det(J)) < 100: raise RuntimeError("calibration: tape_dx / tape_dy move the brick too little or in the same direction (teach them 1-2 cm apart, at right angles)")
        self.cfg["cal"] = {"c0": list(m0["c"]), "a0": m0["angle"], "c_turn": list(meas["turned"]["c"]), "k_roll": k,
                           "J": J.tolist(), "px_per_mm": m0["short"] / self.cfg["short_mm"], "dir": s, "place": P0,
                           "time": datetime.now().isoformat(timespec="seconds"),
                           "turn_offset_mm": round(float(np.hypot(*(np.array(meas["turned"]["c"]) - c0))) / (m0["short"] / self.cfg["short_mm"]), 1)}
        self.C.save_cfg()
        return self.cfg["cal"]

    # ------------------------------------------------------------------ the re-grip
    def target(self, m, action):
        """Measured brick + action -> (grip pose, info). Raises if the brick is outside the calibrated area."""
        cal, P0 = self.cfg["cal"], self.pose("tape_place")
        Dx, Dy = self.pose("tape_dx"), self.pose("tape_dy")
        c, c0 = np.array(m["c"]), np.array(cal["c0"])
        ppm = cal["px_per_mm"]
        if action == "turn":
            turn, ref, want = 90 * cal["dir"], np.array(cal["c_turn"]), c
        else:   # shift along the long axis to the other side of where the fingers were
            turn, ref = 0, c0
            u = np.array([np.cos(np.radians(m["angle"])), np.sin(np.radians(m["angle"]))])
            was = float(np.dot(c0 - c, u))                    # where the fingers were, along the brick (px)
            sh = min(self.cfg["shift_mm"] * ppm, max(0.0, m["long"] / 2 - 8 * ppm))
            want = c - (1 if was >= 0 else -1) * sh * u
        a = np.linalg.solve(np.array(cal["J"]), want - ref)
        if np.abs(a).max() > MAX_BLEND:
            raise RuntimeError(f"brick is {np.hypot(*(c - c0)) / ppm:.0f} mm from the calibrated spot: outside the correction range")
        # small roll correction that lines the fingers up with the brick's sides again (mod 90: the brick was just
        # set down by the gripper, so it is only off by the slip; a long brick stays clamped across its width)
        da = prep.ang_diff(m["angle"], cal["a0"], 90)
        pose = {n: P0[n] + a[0] * (Dx[n] - P0[n]) + a[1] * (Dy[n] - P0[n]) for n in BODY}
        pose["wrist_roll"] = self.fit_roll(P0["wrist_roll"] + (turn + da / cal["k_roll"]) * TICK_PER_DEG, m["sym"])
        return pose, {"action": action, "brick_deg": round(da, 1), "move_mm": round(float(np.hypot(*(c - c0))) / ppm, 1),
                      "blend": [round(float(x), 2) for x in a], "want": want.tolist(), "ref": ref.tolist()}

    def regrip(self, k, g_lock, g_open):
        """k-th re-grip of the brick being inspected (k = 0 first). The brick is held on entry.
        Returns (True, info) holding it again above the spot, or (False, info) with the brick left on the spot."""
        C = self.C
        if k == 0: self.shape = None
        if self.missing():
            ok = self.blind_turn(g_lock, g_open)
            self.last = {"mode": "blind turn (re-grip station not set up: " + "; ".join(self.missing()) + ")", "ok": ok}
            return ok, self.last
        P0, gw = self.pose("tape_place"), self.g_wide(g_open)
        C.say(f"re-grip {k + 1}: set the brick down on the spot")
        self.put(P0, g_lock, g_open)
        info = {}
        for t in range(1, self.cfg["max_tries"] + 1):
            m = self.look(g_open)
            if m is None:
                info = {"ok": False, "why": "brick not seen on the spot (fell off?)"}; break
            self.shape = "square" if m["sym"] == 90 else "long"
            plan = self.plan()
            action = plan[min(k, len(plan) - 1)]
            try:
                pose, info = self.target(m, action)
            except RuntimeError as e:
                info = {"ok": False, "why": str(e)}; break
            f = C.S.frames.get("top")
            if f is not None: self.render(f, m, info["want"], info["ref"])
            info.update(shape=self.shape, attempt=t)
            C.say(f"re-grip {k + 1} ({self.shape}, {action}): brick turned {info['brick_deg']}°, "
                  f"{info['move_mm']} mm off · attempt {t}/{self.cfg['max_tries']}")
            if self.take(pose, g_lock, gw):
                info["ok"] = True; break
            info.update(ok=False, why="not gripped")
            C.goto(dict(C.A.cmd), 0.5, gripper=gw); time.sleep(0.2)     # let go, it is re-measured next try
            C.goto(self.above_of(pose), C.cfg["speed"] * 0.5, gripper=gw)
        self.last = {**info, "k": k, "time": datetime.now().strftime("%H:%M:%S")}
        if not info.get("ok"): C.goto(self.pose("tape_above"), gripper=gw)
        return bool(info.get("ok")), self.last

    def blind_turn(self, g_lock, g_open):
        """Fallback without camera correction: set down, turn the wrist 90 deg, grip again, lift."""
        C, P0 = self.C, self.pose("tape_place")
        self.put(P0, g_lock, g_open)
        turned = {**P0, "wrist_roll": P0["wrist_roll"] + 90 * TICK_PER_DEG * self.roll_dir()}
        C.check()
        return self.take(turned, g_lock, self.g_wide(g_open))
