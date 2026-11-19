"""Simulation test of regrip.py + the phase-2 inspection loop (no hardware, no lerobot needed).
Run: python scripts/sim_regrip.py   (needs numpy, opencv-python(-headless), fastapi)

Sim world: top camera image = grey table + dark L-jig + red brick (rotated rect).
Gripper: roll axis position in the image is linear in (pan, lift) around the spot pose; the held brick sits
D px off the roll axis across the clamp axis (FIXED JAW), plus an offset along the fingers. Grasp succeeds
only if the brick centre is within FINGER px of the gripper along the fingers, fits between the open jaws,
and the clamp axis matches a brick axis (long bricks can only be clamped across their width).
"""
import sys, types, threading, random, math, tempfile, json
from pathlib import Path
import numpy as np, cv2

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "anomaly_capture"
sys.path.insert(0, str(REPO))
# ---- stub lerobot (arm.py imports it at module level)
for m in ("lerobot", "lerobot.motors", "lerobot.motors.feetech", "lerobot.motors.feetech.feetech", "lerobot.motors.motors_bus"):
    sys.modules[m] = types.ModuleType(m)
sys.modules["lerobot.motors.feetech.feetech"].FeetechMotorsBus = object
mb = sys.modules["lerobot.motors.motors_bus"]; mb.Motor = object; mb.MotorNormMode = types.SimpleNamespace(RANGE_M100_100=0)

import collect, regrip, prep
from arm import TICK_PER_DEG as TPD

random.seed(1); np.random.seed(1)
W, H = 1920, 1080
ROI = [800, 400, 1100, 700]
BASE = np.array([950.0, 560.0])      # roll axis in the image at the spot pose
MP = np.array([[0.50, 0.08], [-0.06, 0.45]])   # px per tick of (pan, lift)
D = 14.0                              # fixed jaw: held brick centre is 14 px (~7 mm) off the roll axis
FINGER = 12.0                         # px along the fingers the grasp tolerates
K_SIM, ANG0 = -1.0, 20.0              # image deg per roll deg, image angle of the clamp axis at roll r0
SLIP = (2.0, 2.0)                     # px, deg on release

P0 = {"shoulder_pan": 2000, "shoulder_lift": 2400, "elbow_flex": 2000, "wrist_flex": 2000, "wrist_roll": 1500, "gripper": 1500}
U = {**P0, "shoulder_lift": 1800}
R0 = P0["wrist_roll"]


class World:
    def __init__(self):
        self.held = None          # (along_finger_px, rel_angle_deg) of a held brick
        self.brick = None         # (cx, cy, angle) on the table
        self.size = (66, 66)
        self.log = []

    def axis(self, p):
        return BASE + MP @ np.array([p["shoulder_pan"] - P0["shoulder_pan"], p["shoulder_lift"] - P0["shoulder_lift"]])

    def theta(self, p): return ANG0 + K_SIM * (p["wrist_roll"] - R0) / TPD

    def grip_frame(self, p):
        t = math.radians(self.theta(p)); ax = np.array([math.cos(t), math.sin(t)]); fi = np.array([-ax[1], ax[0]])
        return self.axis(p), ax, fi

    def low(self, p): return abs(p["shoulder_lift"] - P0["shoulder_lift"]) < 200

    def release(self, p):
        a, ax, fi = self.grip_frame(p); along, rel = self.held
        c = a + D * ax + along * fi + np.random.uniform(-SLIP[0], SLIP[0], 2)
        self.brick = (c[0], c[1], self.theta(p) + rel + random.uniform(-SLIP[1], SLIP[1])); self.held = None
        self.log.append(("release", rel))

    def grasp(self, p):
        a, ax, fi = self.grip_frame(p)
        c = np.array(self.brick[:2]); v = c - a
        across, along = float(v @ ax), float(v @ fi)
        rel = (self.brick[2] - self.theta(p) + 180) % 360 - 180
        long_ = self.size[0] > self.size[1] * 1.3
        if long_: ok_ang = abs(((rel - 90) + 90) % 180 - 90) < 12        # long axis must be perpendicular to clamp axis
        else: ok_ang = abs(((rel + 45) % 90) - 45) < 12
        ok = abs(along) < max(self.size) / 2 - 21 and abs(across - D) < 25 and ok_ang   # finger pads stay on the brick
        self.log.append(("grasp", ok, round(along, 1), round(across, 1), round(rel, 1)))
        if ok:
            snap = 90 * round(rel / 90)
            self.held = (along, snap); self.brick = None
        return ok

    def render(self, arm_over):
        img = np.full((H, W, 3), (150, 150, 150), np.uint8)
        cv2.rectangle(img, (880, 470), (890, 660), (60, 60, 60), -1); cv2.rectangle(img, (880, 650), (1060, 660), (60, 60, 60), -1)
        if self.brick:
            cx, cy, ang = self.brick
            pts = cv2.boxPoints(((cx, cy), self.size, ang)).astype(np.int32)
            cv2.fillPoly(img, [pts], (40, 40, 210))
        if arm_over: cv2.circle(img, (950, 560), 120, (30, 30, 30), -1)
        img = np.clip(img.astype(np.int16) + np.random.randint(-3, 4, img.shape), 0, 255).astype(np.uint8)
        return img


WORLD = World()


class FakeArm:
    def __init__(self):
        self.fl = {n: (0, 4095) for n in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]}
        self.fl["gripper"] = (1000, 3000)
        self.cmd = {k: float(v) for k, v in U.items()}; self.pos = dict(self.cmd); self.goal = dict(self.cmd)
        self.stop = threading.Event(); self.mode = "hold"; self.running = False; self.io = threading.Lock()
        self.f = True
        self.cmd["gripper"] = 1500.0

    def set_mode(self, m, force=False): self.mode = m

    def goto(self, tgt, speed):
        prev_g = self.cmd["gripper"]
        self.cmd = dict(tgt); self.goal = dict(tgt)
        g = tgt["gripper"]
        if WORLD.held and g > prev_g + 100 and WORLD.low(tgt): WORLD.release(tgt)
        elif not WORLD.held and WORLD.brick and g < prev_g - 100 and WORLD.low(tgt): WORLD.grasp(tgt)
        self.pos = dict(tgt)
        if WORLD.held: self.pos["gripper"] = g + 150      # stalled on the brick
        return 0


class FakeS:
    live = ["top", "front"]
    def __init__(self, A): self.A = A; self.frames = {}
    def averaged(self, cams):
        over = abs(self.A.cmd["shoulder_pan"] - CLEAR["shoulder_pan"]) > 5
        img = WORLD.render(over); self.frames["top"] = img
        return {k: (img, img, img) for k in cams if k == "top"} | ({"front": (img, img, img)} if "front" in cams else {})


CLEAR = {**U, "shoulder_pan": 2900, "shoulder_lift": 1500}
ROUT = {"home": {**U, "gripper": 2400}, "wp1": U, "tape_above": U, "tape_place": P0,
        "tape_dx": {**P0, "shoulder_pan": P0["shoulder_pan"] + 60},
        "tape_dy": {**P0, "shoulder_lift": P0["shoulder_lift"] + 60, "elbow_flex": P0["elbow_flex"] - 40},
        "regrip_clear": CLEAR}


def make():
    A = FakeArm(); S = FakeS(A)
    ctx = types.SimpleNamespace(R={"routines": {k: {"waypoints": [{"name": k, "pos": v, "capture": False}]} for k, v in ROUT.items()}})
    ctx.R["routines"]["wp2"] = {"waypoints": [{"name": "wp2-1", "pos": {**U, "shoulder_pan": 1200}, "capture": True},
                                              {"name": "wp2-2", "pos": {**U, "shoulder_pan": 1250}, "capture": True}]}
    shots = []
    def capture(cams, label, brick, view, cycle, run): shots.append(view); ctx.R["last_files"] = [view]; return 1
    ctx.capture = capture
    tmp = Path(tempfile.mkdtemp())
    C = collect.Collector.__new__(collect.Collector)
    C.S, C.A, C.ctx, C.DATA = S, A, ctx, tmp
    C.cfg_path = tmp / "collect_settings.json"
    C.cfg = {"home": "home", "wp1": "wp1", "wp2": "wp2", "drops": "drops", "phase2": True, "regrasp": True, "speed": 0.5, "roll_jitter": 45}
    C.state, C.msg, C.ok, C.info = "idle", "", True, {}
    C.active = False; C.stop_flag = threading.Event(); C.last_verdict = None; C.session = "sim"
    C.rg = regrip.Regrip(C)
    return C, A, shots


def faces(rel): return "A" if round(rel / 90) % 2 == 0 else "B"


ok_all = True
def check(cond, msg):
    global ok_all
    print(("PASS " if cond else "FAIL ") + msg); ok_all &= bool(cond)


# ---------------- 1. setup + calibration with a centred square brick
C, A, shots = make()
C.cfg["regrip"]["roi"] = ROI
WORLD.held, WORLD.brick = None, None
A.cmd.update(CLEAR); C.rg.save_bg()
check(C.rg.measure() is None, "empty spot: nothing detected (reference works, jig ignored)")
WORLD.held = (0.0, 0.0)
A.cmd.update(U); A.cmd["gripper"] = 1500.0; A.pos = dict(A.cmd); A.pos["gripper"] = 1650
check(C.holding(), "sim: brick held at start")
print("missing before cal:", C.rg.missing())
C.test_wp2("calibrate")
print("calibrate ->", C.state, "|", C.msg)
cal = C.cfg["regrip"]["cal"]
check(cal is not None and C.state == "done", "calibration finished")
if cal:
    check(cal["k_roll"] == K_SIM, f"roll gain {cal['k_roll']:.2f} ~ {K_SIM}")
    Jt = MP @ np.diag([60, 60])
    check(np.allclose(np.array(cal["J"]), Jt, atol=4), f"J {np.round(cal['J'], 1).tolist()} ~ {Jt.tolist()}")
    check(cal["turn_offset_mm"] > 5, f"turned grip point moves {cal['turn_offset_mm']} mm (fixed jaw is modelled)")

# ---------------- 2. square brick held OFF-centre (like an ACT pick): camera-corrected turn
n_ok = 0; N = 20
for i in range(N):
    WORLD.held = (random.uniform(-9, 9), 0.0); WORLD.brick = None
    rel0 = WORLD.held[1]
    A.cmd.update(U); A.cmd["gripper"] = 1500.0; A.pos = dict(A.cmd); A.pos["gripper"] = 1650
    ok, info = C.rg.regrip(0, 1500.0, 2400.0)
    if ok and WORLD.held and faces(WORLD.held[1]) != faces(rel0): n_ok += 1
    elif i < 3: print("  fail:", info, WORLD.log[-3:])
check(n_ok >= N - 1, f"square re-grip turned 90° (other 2 faces free): {n_ok}/{N}")

# ---------------- 3. blind fallback (old behaviour) for comparison
C2, A2, _ = make()
n_blind = 0
for i in range(N):
    WORLD.held = (random.uniform(-9, 9), 0.0); WORLD.brick = None
    A2.cmd.update(U); A2.cmd["gripper"] = 1500.0; A2.pos = dict(A2.cmd); A2.pos["gripper"] = 1650
    ok, info = C2.rg.regrip(0, 1500.0, 2400.0)
    n_blind += bool(ok)
print(f"INFO blind 90° turn (no camera): {n_blind}/{N} (fixed-jaw offset makes it unreliable)")

# ---------------- 4. long 2x4 brick: shift plan
WORLD.size = (132, 66)
n_ok = 0
for i in range(N):
    along0 = random.choice([-1, 1]) * random.uniform(4, 9)
    WORLD.held = (along0, 90.0); WORLD.brick = None     # long axis along the fingers, clamped across the width
    A.cmd.update(U); A.cmd["gripper"] = 1500.0; A.pos = dict(A.cmd); A.pos["gripper"] = 1650
    ok, info = C.rg.regrip(0, 1500.0, 2400.0)
    if ok and WORLD.held:
        moved = WORLD.held[0] - along0
        # finger strip moved by >= ~shift (18 mm ~ 37 px) to the other side
        if abs(moved) > 25 and np.sign(WORLD.held[0]) != np.sign(along0): n_ok += 1
        elif i < 3: print("  shift small:", round(along0, 1), "->", round(WORLD.held[0], 1), info)
    elif i < 3: print("  fail:", info, WORLD.log[-3:])
check(C.rg.shape == "long", "long brick recognised from the footprint")
check(n_ok >= N - 1, f"long brick re-gripped further along on the other side: {n_ok}/{N}")

# ---------------- 5. inspection loop: early stop on a cross, full plan without a model
WORLD.size = (66, 66)
def run_phase2(verdict_fn):
    shots.clear(); WORLD.held = (3.0, 0.0); WORLD.brick = None
    A.cmd.update(U); A.cmd["gripper"] = 1500.0; A.pos = dict(A.cmd); A.pos["gripper"] = 1650
    C.verdict = verdict_fn
    wp1 = U; wp2 = C.wps("wp2")
    n, on_tape = C.phase2(wp1, wp2, 1500.0, 2400.0, "good", "red", 1)
    return n, on_tape, list(shots), dict(C.last_verdict)

n, tape, sh, v = run_phase2(lambda files: None)
print("no model:", sh, v)
check(sh == ["wp2-1", "wp2-2", "wp2-1_g2", "wp2-2_g2"] and v["grips"] == 2 and not tape, "no model: grip 1 + one re-grip, all views shot")
n, tape, sh, v = run_phase2(lambda files: "defect" if files == ["wp2-1"] else "good")
print("cross on first view:", sh, v)
check(sh == ["wp2-1"] and v["result"] == "defect" and v["grips"] == 1, "cross in grip 1: stops at once, no re-grip")
n, tape, sh, v = run_phase2(lambda files: "defect" if files == ["wp2-2_g2"] else "good")
print("cross under the fingers:", sh, v)
check(v["result"] == "defect" and sh[-1] == "wp2-2_g2", "cross under the first grip found after the re-grip")
n, tape, sh, v = run_phase2(lambda files: "good")
check(v["result"] == "good", "clean brick: good after all faces")
n, tape, sh, v = run_phase2(lambda files: "unsure" if files == ["wp2-1"] else "good")
check(v["result"].startswith("defect (unsure"), "unsure view -> fail-safe defect")

print("\nALL PASS" if ok_all else "\nSOME FAILED")
sys.exit(0 if ok_all else 1)
