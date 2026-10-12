import json
import math
import os
import random
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=1,
    min_hand_detection_confidence=0.6,
)
detector = vision.HandLandmarker.create_from_options(options)

CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),
               (10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),
               (17,18),(18,19),(19,20),(0,17)]
WHITE = (255, 255, 255)
GREY = (200, 200, 200)

# ============================================================ high scores
SCORE_FILE = "highscores.json"


def load_best():
    try:
        with open(SCORE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


BEST = load_best()


def record(key, score):
    """Save a new best score. Returns True if it's a record."""
    if score > BEST.get(key, 0):
        BEST[key] = score
        try:
            with open(SCORE_FILE, "w") as f:
                json.dump(BEST, f)
        except Exception:
            pass
        return True
    return False


def finish(s, key, score):
    if not s.get("saved"):
        s["saved"] = True
        s["new_best"] = record(key, score)


# ================================================================ helpers
def fingers_up(lm):
    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
    return [lm[t].y < lm[p].y for t, p in zip(tips, pips)]


def inside(pt, rect):
    if pt is None:
        return False
    x1, y1, x2, y2 = rect
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2


def darken(frame, k):
    return (frame * k).astype(np.uint8)


def draw_skeleton(frame, pts, color=(110, 110, 110)):
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 1)


def put_center(frame, text, y, scale=1.0, color=WHITE, thick=2):
    tw = cv2.getTextSize(text, 0, scale, thick)[0][0]
    cv2.putText(frame, text, ((frame.shape[1] - tw) // 2, y), 0, scale, color, thick)


def draw_button(frame, rect, label, color, hover):
    x1, y1, x2, y2 = rect
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    alpha = 0.65 if hover else 0.25
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4 if hover else 2)
    tw, th = cv2.getTextSize(label, 0, 0.8, 2)[0]
    cv2.putText(frame, label, (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2 - 8),
                0, 0.8, WHITE, 2)


def seg_dist(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = dx * dx + dy * dy
    if length == 0:
        return math.dist(p, a)
    t = max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length))
    return math.dist(p, (a[0] + t * dx, a[1] + t * dy))


def burst(s, x, y, color, n=16):
    s.setdefault("parts", [])
    for _ in range(n):
        ang = random.uniform(0, 2 * math.pi)
        spd = random.uniform(2, 8)
        s["parts"].append({"x": x, "y": y, "vx": math.cos(ang) * spd,
                           "vy": math.sin(ang) * spd, "life": 25, "col": color})


def draw_particles(frame, s):
    for p in s.get("parts", []):
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["vy"] += 0.3
        p["life"] -= 1
        cv2.circle(frame, (int(p["x"]), int(p["y"])), 4, p["col"], -1)
    s["parts"] = [p for p in s.get("parts", []) if p["life"] > 0]


def hud(frame, s, key, score, hint):
    h = frame.shape[0]
    cv2.putText(frame, f"Score: {score}", (20, 40), 0, 1.1, WHITE, 3)
    cv2.putText(frame, f"Best: {BEST.get(key, 0)}", (20, 75), 0, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, hint, (20, h - 20), 0, 0.55, GREY, 1)


def game_over(frame, s, key, score):
    h = frame.shape[0]
    finish(s, key, score)
    put_center(frame, "GAME OVER", h // 2 - 30, 2, (0, 0, 255), 5)
    put_center(frame, f"Score {score}    Best {BEST.get(key, 0)}", h // 2 + 25, 1.0)
    if s.get("new_best"):
        put_center(frame, "NEW BEST!", h // 2 + 70, 1.2, (0, 255, 255), 3)
    put_center(frame, "Pinch anywhere to play again", h // 2 + 115, 0.8)


def update_open(s, hd):
    s["hist"].append(sum(fingers_up(hd["lm"])) >= 3)
    s["opened"] = sum(s["hist"]) > len(s["hist"]) / 2


# ======================================================= GAME 1: FRUIT SLICER
FRUIT_COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (0, 255, 255), (255, 0, 255)]


def init_fruit(w, h):
    return {"items": [], "parts": [], "trail": deque(maxlen=8),
            "score": 0, "lives": 5, "frame": 0, "over": False}


def run_fruit(frame, ctx, s):
    w, h, hands = ctx["w"], ctx["h"], ctx["hands"]
    G = 0.5
    tip = hands[0]["index"] if hands else None
    if tip is not None:
        s["trail"].append(tip)
    else:
        s["trail"].clear()

    if not s["over"]:
        s["frame"] += 1
        spawn_every = max(20, 50 - s["score"])
        if s["frame"] % spawn_every == 0:
            s["items"].append({
                "x": random.randint(int(w * 0.2), int(w * 0.8)), "y": h + 30,
                "vx": random.uniform(-3, 3),
                "vy": -random.uniform(0.75, 0.95) * math.sqrt(2 * G * h),
                "r": random.randint(28, 38), "bomb": random.random() < 0.2,
                "col": random.choice(FRUIT_COLORS)})

        fast = len(s["trail"]) >= 2 and math.dist(s["trail"][-1], s["trail"][-2]) > 18
        remaining = []
        for it in s["items"]:
            it["x"] += it["vx"]
            it["vy"] += G
            it["y"] += it["vy"]
            if fast and seg_dist((it["x"], it["y"]), s["trail"][-2], s["trail"][-1]) < it["r"] + 10:
                if it["bomb"]:
                    s["lives"] -= 1
                    burst(s, it["x"], it["y"], (0, 0, 255))
                else:
                    s["score"] += 1
                    burst(s, it["x"], it["y"], it["col"])
                continue
            if it["y"] > h + it["r"] and it["vy"] > 0:
                if not it["bomb"]:
                    s["lives"] -= 1
                continue
            remaining.append(it)
        s["items"] = remaining
        if s["lives"] <= 0:
            s["over"] = True

    draw_particles(frame, s)
    for it in s["items"]:
        c = (int(it["x"]), int(it["y"]))
        if it["bomb"]:
            cv2.circle(frame, c, it["r"], (50, 50, 50), -1)
            cv2.line(frame, (c[0] - 10, c[1] - 10), (c[0] + 10, c[1] + 10), (0, 0, 255), 4)
            cv2.line(frame, (c[0] + 10, c[1] - 10), (c[0] - 10, c[1] + 10), (0, 0, 255), 4)
        else:
            cv2.circle(frame, c, it["r"], it["col"], -1)
        cv2.circle(frame, c, it["r"], WHITE, 2)

    pts = list(s["trail"])
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (255, 255, 0), max(1, i))

    hud(frame, s, "fruit", s["score"], "Swipe FAST with your index finger. Avoid the bombs!")
    cv2.putText(frame, f"Lives: {s['lives']}", (20, 110), 0, 0.9, (0, 0, 255), 2)
    if s["over"]:
        game_over(frame, s, "fruit", s["score"])


# ============================================================ GAME 2: SNAKE
SP = 4   # frames between body segments


def rand_food(w, h):
    return (random.randint(40, w - 40), random.randint(110, h - 60))


def init_snake(w, h):
    return {"head": [w / 2, h / 2], "path": deque([(w / 2, h / 2)], maxlen=6 * SP + 1),
            "score": 0, "over": False, "food": rand_food(w, h), "parts": []}


def run_snake(frame, ctx, s):
    w, h, hands = ctx["w"], ctx["h"], ctx["hands"]

    if not s["over"] and hands:
        tip = hands[0]["index"]
        hx, hy = s["head"]
        dx, dy = tip[0] - hx, tip[1] - hy
        d = math.hypot(dx, dy)
        speed = min(8 + s["score"] * 0.4, 15)
        if d > 3:
            step = min(d, speed)
            hx = min(max(hx + dx / d * step, 10), w - 10)
            hy = min(max(hy + dy / d * step, 10), h - 10)
            s["head"] = [hx, hy]
            s["path"].appendleft((hx, hy))

        segs = list(s["path"])[::SP]
        if len(segs) > 8:
            for sg in segs[8:]:
                if math.dist((hx, hy), sg) < 13:
                    s["over"] = True
                    burst(s, hx, hy, (0, 255, 0), 30)
                    break

        if math.dist((hx, hy), s["food"]) < 28:
            s["score"] += 1
            burst(s, s["food"][0], s["food"][1], (0, 0, 255))
            s["food"] = rand_food(w, h)
            s["path"] = deque(s["path"], maxlen=(6 + 2 * s["score"]) * SP + 1)

    segs = list(s["path"])[::SP]
    n = len(segs)
    for i in range(n - 1, -1, -1):
        t = i / max(1, n)
        col = (int(180 * (1 - t)), 255 - int(110 * t), int(60 * t))
        cv2.circle(frame, (int(segs[i][0]), int(segs[i][1])), int(15 - 7 * t), col, -1)

    hx, hy = int(s["head"][0]), int(s["head"][1])
    cv2.circle(frame, (hx, hy), 16, (0, 255, 120), -1)
    cv2.circle(frame, (hx - 5, hy - 4), 4, WHITE, -1)
    cv2.circle(frame, (hx + 5, hy - 4), 4, WHITE, -1)
    cv2.circle(frame, (hx - 5, hy - 4), 2, (0, 0, 0), -1)
    cv2.circle(frame, (hx + 5, hy - 4), 2, (0, 0, 0), -1)

    pulse = int(3 * math.sin(time.time() * 6))
    cv2.circle(frame, (int(s["food"][0]), int(s["food"][1])), 13 + pulse, (0, 0, 255), -1)
    cv2.circle(frame, (int(s["food"][0]), int(s["food"][1])), 13 + pulse, WHITE, 2)

    draw_particles(frame, s)
    hud(frame, s, "snake", s["score"], "Guide the snake with your index finger. Eat, grow, don't bite yourself!")
    if not hands and not s["over"]:
        put_center(frame, "Show your hand to move", h // 2, 1.1)
    if s["over"]:
        game_over(frame, s, "snake", s["score"])


# ========================================================= GAME 3: SKY FLAP
def init_flap(w, h):
    return {"y": h / 2, "r": 24.0, "pipes": [], "score": 0, "over": False,
            "hist": deque(maxlen=5), "opened": True, "parts": []}


def run_flap(frame, ctx, s):
    w, h, hands = ctx["w"], ctx["h"], ctx["hands"]
    bx = int(w * 0.25)
    HALF = 35

    if hands:
        update_open(s, hands[0])
        s["y"] += (hands[0]["pts"][9][1] - s["y"]) * 0.4
    s["y"] = min(max(s["y"], 20), h - 20)
    s["r"] += ((24 if s["opened"] else 9) - s["r"]) * 0.4
    r, y = s["r"], s["y"]

    if not s["over"] and hands:
        speed = min(5 + s["score"] * 0.15, 10)
        if not s["pipes"] or s["pipes"][-1]["x"] < w - 340:
            narrow = s["score"] >= 2 and random.random() < 0.35
            gap = 52 if narrow else 150
            s["pipes"].append({"x": w + 50, "gap": gap, "narrow": narrow, "passed": False,
                               "gy": random.randint(gap // 2 + 40, h - gap // 2 - 40)})
        for p in s["pipes"]:
            p["x"] -= speed
            if p["x"] - HALF < bx + r and bx - r < p["x"] + HALF:
                if y - r < p["gy"] - p["gap"] / 2 or y + r > p["gy"] + p["gap"] / 2:
                    s["over"] = True
                    burst(s, bx, y, (0, 255, 255), 30)
            if not p["passed"] and p["x"] + HALF < bx - r:
                p["passed"] = True
                s["score"] += 1
        s["pipes"] = [p for p in s["pipes"] if p["x"] > -80]

    for p in s["pipes"]:
        col = (0, 0, 255) if p["narrow"] else (0, 200, 0)
        x, top, bot = int(p["x"]), int(p["gy"] - p["gap"] / 2), int(p["gy"] + p["gap"] / 2)
        cv2.rectangle(frame, (x - HALF, 0), (x + HALF, top), col, -1)
        cv2.rectangle(frame, (x - HALF - 6, top - 22), (x + HALF + 6, top), col, -1)
        cv2.rectangle(frame, (x - HALF, bot), (x + HALF, h), col, -1)
        cv2.rectangle(frame, (x - HALF - 6, bot), (x + HALF + 6, bot + 22), col, -1)
        cv2.rectangle(frame, (x - HALF, 0), (x + HALF, top), WHITE, 2)
        cv2.rectangle(frame, (x - HALF, bot), (x + HALF, h), WHITE, 2)

    if not s["over"]:
        c = (bx, int(y))
        cv2.circle(frame, c, int(r), (0, 220, 255), -1)
        cv2.circle(frame, c, int(r), WHITE, 2)
        cv2.circle(frame, (c[0] + int(r * 0.35), c[1] - int(r * 0.3)), max(2, int(r * 0.2)), (0, 0, 0), -1)
        cv2.line(frame, (c[0] + int(r), c[1]), (c[0] + int(r) + 10, c[1] + 3), (0, 120, 255), 4)

    draw_particles(frame, s)
    hud(frame, s, "flap", s["score"], "Hand height = bird height | RED gap = make a FIST to shrink!")
    if not hands and not s["over"]:
        put_center(frame, "Show your hand to fly", h // 2, 1.1)
    if s["over"]:
        game_over(frame, s, "flap", s["score"])


# ====================================================== GAME 4: STAR DEFENDER
def init_space(w, h):
    return {"x": w / 2, "bullets": [], "enemies": [], "energy": 100.0, "lives": 3,
            "score": 0, "frame": 0, "over": False, "cool": 0,
            "hist": deque(maxlen=5), "opened": True, "parts": []}


def run_space(frame, ctx, s):
    w, h, hands = ctx["w"], ctx["h"], ctx["hands"]
    sy = h - 90

    if hands:
        update_open(s, hands[0])
        s["x"] += (hands[0]["pts"][9][0] - s["x"]) * 0.5
    s["x"] = min(max(s["x"], 40), w - 40)
    shield = bool(hands) and (not s["opened"]) and s["energy"] > 0

    if not s["over"] and hands:
        s["frame"] += 1
        s["energy"] = max(0, s["energy"] - 1.2) if shield else min(100, s["energy"] + 0.4)

        s["cool"] -= 1
        if s["opened"] and s["cool"] <= 0:
            s["bullets"].append([s["x"], sy - 30])
            s["cool"] = 8

        if s["frame"] % max(14, 45 - s["score"] // 2) == 0:
            s["enemies"].append({"x0": random.randint(60, w - 60), "x": 0, "y": -30,
                                 "vy": random.uniform(2, 3.5) + min(s["score"] * 0.04, 3),
                                 "ph": random.uniform(0, 6.28), "amp": random.randint(0, 50)})

        for b in s["bullets"]:
            b[1] -= 16

        alive = []
        for e in s["enemies"]:
            e["y"] += e["vy"]
            e["x"] = min(max(e["x0"] + e["amp"] * math.sin(e["ph"] + s["frame"] * 0.05), 25), w - 25)
            hit = False
            for b in s["bullets"]:
                if math.dist((e["x"], e["y"]), (b[0], b[1])) < 28:
                    b[1] = -100
                    hit = True
                    break
            if hit:
                s["score"] += 1
                burst(s, e["x"], e["y"], (0, 165, 255))
                continue
            if math.dist((e["x"], e["y"]), (s["x"], sy)) < 42:
                if shield:
                    s["score"] += 1
                    burst(s, e["x"], e["y"], (255, 200, 0))
                else:
                    s["lives"] -= 1
                    burst(s, e["x"], e["y"], (0, 0, 255))
                continue
            if e["y"] > h + 30:
                s["lives"] -= 1
                continue
            alive.append(e)
        s["enemies"] = alive
        s["bullets"] = [b for b in s["bullets"] if b[1] > -10]

        if s["lives"] <= 0:
            s["over"] = True
            burst(s, s["x"], sy, (0, 255, 255), 40)

    for e in s["enemies"]:
        c = (int(e["x"]), int(e["y"]))
        cv2.circle(frame, c, 22, (0, 0, 255), -1)
        cv2.circle(frame, c, 22, WHITE, 2)
        cv2.circle(frame, (c[0] - 8, c[1] - 3), 4, WHITE, -1)
        cv2.circle(frame, (c[0] + 8, c[1] - 3), 4, WHITE, -1)
    for b in s["bullets"]:
        cv2.rectangle(frame, (int(b[0]) - 3, int(b[1]) - 12), (int(b[0]) + 3, int(b[1])), (255, 255, 0), -1)

    if not s["over"]:
        x = int(s["x"])
        ship = np.array([(x, sy - 28), (x - 24, sy + 22), (x, sy + 10), (x + 24, sy + 22)], np.int32)
        cv2.fillPoly(frame, [ship], (255, 220, 0))
        cv2.polylines(frame, [ship], True, WHITE, 2)
        if shield:
            cv2.circle(frame, (x, sy), 50, (255, 200, 0), 4)

    cv2.rectangle(frame, (w - 230, h - 40), (w - 20, h - 20), WHITE, 2)
    cv2.rectangle(frame, (w - 228, h - 38), (w - 228 + int(206 * s["energy"] / 100), h - 22),
                  (255, 200, 0), -1)
    cv2.putText(frame, "SHIELD", (w - 230, h - 48), 0, 0.5, WHITE, 1)

    draw_particles(frame, s)
    hud(frame, s, "space", s["score"], "Move hand = steer | OPEN = auto-fire | FIST = shield (uses energy)")
    cv2.putText(frame, f"Lives: {s['lives']}", (20, 110), 0, 0.9, (0, 0, 255), 2)
    if not hands and not s["over"]:
        put_center(frame, "Show your hand to play", h // 2, 1.1)
    if s["over"]:
        game_over(frame, s, "space", s["score"])


# ============================================================ game registry
# To add a game: write init_xxx + run_xxx, then add one line below.
GAMES = {
    "fruit": {"label": "FRUIT SLICER",  "init": init_fruit, "run": run_fruit, "color": (0, 160, 255)},
    "snake": {"label": "FINGER SNAKE",  "init": init_snake, "run": run_snake, "color": (0, 200, 0)},
    "flap":  {"label": "SKY FLAP",      "init": init_flap,  "run": run_flap,  "color": (255, 150, 0)},
    "space": {"label": "STAR DEFENDER", "init": init_space, "run": run_space, "color": (200, 0, 200)},
}
ORDER = ["fruit", "snake", "flap", "space"]

app = {"scene": "menu", "cooldown": 0}
st = {}
prev_pinch = False


def enter(name, w, h):
    st.clear()
    app["scene"] = name
    app["cooldown"] = 15
    if name != "menu":
        st.update(GAMES[name]["init"](w, h))


# ================================================================ main loop
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    hands = []
    for lm in result.hand_landmarks:
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        size = math.dist(pts[0], pts[9])
        thumb, index = pts[4], pts[8]
        mid = ((thumb[0] + index[0]) // 2, (thumb[1] + index[1]) // 2)
        hands.append({"lm": lm, "pts": pts, "index": index, "mid": mid,
                      "pinch": math.dist(thumb, index) < 0.32 * size})

    # a "click" is the moment a pinch starts
    click_pt = next((hd["mid"] for hd in hands if hd["pinch"]), None)
    clicked = click_pt is not None and not prev_pinch and app["cooldown"] == 0
    prev_pinch = click_pt is not None
    if app["cooldown"] > 0:
        app["cooldown"] -= 1
    cursor = click_pt if click_pt is not None else (hands[0]["mid"] if hands else None)

    scene = app["scene"]
    frame = darken(frame, 0.35 if scene == "menu" else 0.75)
    for hd in hands:
        draw_skeleton(frame, hd["pts"])

    if scene == "menu":
        put_center(frame, "GESTURE ARCADE", 70, 1.4, WHITE, 3)
        for i, key in enumerate(ORDER):
            col, row = i % 2, i // 2
            x1 = w // 2 - 300 if col == 0 else w // 2 + 10
            rect = (x1, 130 + row * 130, x1 + 290, 130 + row * 130 + 100)
            hover = inside(cursor, rect)
            draw_button(frame, rect, GAMES[key]["label"], GAMES[key]["color"], hover)
            cv2.putText(frame, f"BEST: {BEST.get(key, 0)}", (rect[0] + 12, rect[3] - 10),
                        0, 0.55, (0, 255, 255), 1)
            if hover and clicked:
                enter(key, w, h)
        put_center(frame, "Pinch (thumb + index) on a game to play it", h - 25, 0.6, GREY, 1)
    else:
        GAMES[scene]["run"](frame, {"w": w, "h": h, "hands": hands}, st)

        back = (w - 130, 12, w - 12, 62)
        draw_button(frame, back, "BACK", (80, 80, 255), inside(cursor, back))
        if clicked and inside(click_pt, back):
            enter("menu", w, h)
        elif clicked and st.get("over"):       # pinch anywhere to play again
            enter(scene, w, h)

    if hands:
        cv2.circle(frame, hands[0]["mid"], 9, (0, 255, 0) if hands[0]["pinch"] else WHITE, 2)

    cv2.imshow("Gesture Arcade", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:                                   # Esc = quit
        break
    if key == ord("m"):                             # M = menu
        enter("menu", w, h)
    if key == ord("r") and app["scene"] != "menu":  # R = restart current game
        enter(app["scene"], w, h)

cap.release()
cv2.destroyAllWindows()