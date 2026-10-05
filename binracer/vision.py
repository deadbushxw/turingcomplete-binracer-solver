# -*- coding: utf-8 -*-
"""Vision layer for the Turing Complete "Binary Racer" (二进制速算) level.

Everything works in *frame space*: frames are the 1568x980 window snapshots
(the game window is a 2560x1600 borderless fullscreen window; scale = 1568/2560 = 0.6125).
Use to_physical() when turning a frame-space position into a real screen position.

Recognition strategy
--------------------
The game renders digits with NoroshiCode_Bold.ttf, which ships with the game.
Rendering the same digits offline and matching them (normalised cross correlation)
reproduces the on-screen glyphs almost exactly (NCC 0.95-0.98), so recognition is
template matching against *rendered* glyphs rather than OCR.
"""
from __future__ import annotations

import os
import re

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_FONT = os.path.join(os.path.dirname(HERE), "assets", "font", "NoroshiCode_Bold.ttf")
FONT_NAME = "NoroshiCode_Bold.ttf"
GAME_FONT_REL = os.path.join("asset", "font", FONT_NAME)
FONT_PATH = LOCAL_FONT          # refreshed by font_path() the first time it is needed


def _steam_roots():
    """Steam install roots, discovered instead of hard-coded (registry + library files)."""
    roots = []
    try:
        import winreg
        for hive, key, value in ((winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
                                 (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
                                 (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam", "InstallPath")):
            try:
                with winreg.OpenKey(hive, key) as k:
                    p = winreg.QueryValueEx(k, value)[0]
                    if p:
                        roots.append(str(p))
            except OSError:
                pass
    except ImportError:
        pass
    for base in roots[:]:
        vdf = os.path.join(base, "steamapps", "libraryfolders.vdf")
        try:
            with open(vdf, "r", encoding="utf-8", errors="ignore") as fh:
                for m in re.findall(r'"path"\s*"([^"]+)"', fh.read()):
                    roots.append(m.replace("\\\\", "\\"))
        except OSError:
            pass
    roots += [r"C:\Program Files (x86)\Steam", r"C:\Program Files\Steam"]
    seen, uniq = set(), []
    for r in roots:
        r = os.path.normpath(r)
        if r.lower() not in seen and os.path.isdir(r):
            seen.add(r.lower())
            uniq.append(r)
    return uniq


def font_path(force=None):
    """Locate the game font without assuming any machine-specific location.

    Order: explicit path -> $TC_BINRACER_FONT -> local assets copy -> Steam library scan.
    """
    global FONT_PATH
    if force:
        FONT_PATH = force
    for cand in (FONT_PATH, os.environ.get("TC_BINRACER_FONT"), LOCAL_FONT):
        if cand and os.path.isfile(cand):
            FONT_PATH = cand
            return FONT_PATH
    for root in _steam_roots():
        p = os.path.join(root, "steamapps", "common", "Turing Complete", GAME_FONT_REL)
        if os.path.isfile(p):
            FONT_PATH = p
            return FONT_PATH
    raise FileNotFoundError(
        "找不到游戏字体 %s。请任选一种方式:\n"
        "  1) 设置环境变量 TC_BINRACER_FONT=<游戏的 asset/font/%s 完整路径>\n"
        "  2) 把该字体复制到 %s\n"
        "  3) 用 --font <路径> 启动" % (FONT_NAME, FONT_NAME, LOCAL_FONT))

BITS = (128, 64, 32, 16, 8, 4, 2, 1)
FRAME_W, FRAME_H = 1568, 980
SCREEN_W, SCREEN_H = 2560, 1600
SCALE = FRAME_W / float(SCREEN_W)          # 0.6125

QUESTION_SIZES = tuple(range(44, 58))       # question digits ~ size 50
SUM_SIZES = tuple(range(40, 54))            # running sum ~ size 46

_TEMPLATE_CACHE: dict[tuple[int, int], np.ndarray] = {}


# --------------------------------------------------------------------------- helpers
def to_physical(x: float, y: float) -> tuple[int, int]:
    return int(round(x / SCALE)), int(round(y / SCALE))


def to_frame(x: float, y: float) -> tuple[int, int]:
    return int(round(x * SCALE)), int(round(y * SCALE))


def hsv(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def _mask(img: np.ndarray, lo, hi) -> np.ndarray:
    return cv2.inRange(hsv(img), np.array(lo), np.array(hi))


def orange_mask(img):   # question digits / sum digits / buttons
    return _mask(img, (15, 110, 170), (38, 255, 255))


def pink_mask(img):     # bit chip, off
    return cv2.bitwise_or(_mask(img, (160, 80, 120), (180, 255, 255)),
                          _mask(img, (0, 80, 120), (8, 255, 255)))


def green_mask(img):    # bit chip, on
    return _mask(img, (40, 80, 80), (85, 255, 255))


def red_mask(img):      # "时间到"
    return cv2.bitwise_or(_mask(img, (0, 120, 120), (10, 255, 255)),
                          _mask(img, (170, 120, 120), (180, 255, 255)))


def blue_mask(img):    # dialog title bar / 继续 button
    return _mask(img, (100, 60, 70), (140, 255, 255))


def white_mask(img):
    return _mask(img, (0, 0, 175), (180, 60, 255))


def blobs(mask: np.ndarray, min_area: int = 200, roi=None) -> list[dict]:
    """Connected components of a binary mask, as dicts with bbox/centre/area."""
    if roi is not None:
        x0, y0, x1, y1 = [int(v) for v in roi]
        sub = mask[y0:y1, x0:x1]
    else:
        x0 = y0 = 0
        sub = mask
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(sub, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        out.append(dict(x=x + x0, y=y + y0, w=w, h=h, area=area,
                        cx=x + x0 + w / 2.0, cy=y + y0 + h / 2.0))
    return sorted(out, key=lambda b: b["x"])


# --------------------------------------------------------------------------- glyph matching
def render_digit(digit: int, size: int) -> np.ndarray:
    key = (digit, size)
    cached = _TEMPLATE_CACHE.get(key)
    if cached is not None:
        return cached
    font = ImageFont.truetype(font_path(), size)
    im = Image.new("L", (160, 160), 0)
    ImageDraw.Draw(im).text((30, 30), str(digit), 255, font=font)
    a = np.array(im)
    ys, xs = np.where(a > 100)
    m = (a[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 100).astype(np.uint8) * 255
    _TEMPLATE_CACHE[key] = m
    return m


def render_text(text: str, size: int) -> np.ndarray:
    font = ImageFont.truetype(FONT_PATH, size)
    im = Image.new("L", (60 * len(text) + 120, 200), 0)
    ImageDraw.Draw(im).text((30, 40), text, 255, font=font)
    a = np.array(im)
    ys, xs = np.where(a > 100)
    return (a[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 100).astype(np.uint8) * 255


def _norm_height(m: np.ndarray, height: int = 48) -> np.ndarray:
    h, w = m.shape
    if h == 0 or w == 0:
        return np.zeros((height, 1), np.float32)
    nw = max(1, int(round(w * height / float(h))))
    return cv2.resize(m.astype(np.float32), (nw, height), interpolation=cv2.INTER_AREA)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    A = _norm_height(a)
    B = _norm_height(b)
    width = max(A.shape[1], B.shape[1])
    PA = np.zeros((A.shape[0], width), np.float32); PA[:, :A.shape[1]] = A
    PB = np.zeros((B.shape[0], width), np.float32); PB[:, :B.shape[1]] = B
    PA -= PA.mean(); PB -= PB.mean()
    d = float(np.sqrt((PA * PA).sum()) * np.sqrt((PB * PB).sum()))
    return float((PA * PB).sum() / d) if d else 0.0


def classify_glyph(glyph: np.ndarray, sizes=QUESTION_SIZES, digits=range(10)):
    """Return (digit, score, margin) for a tight-cropped glyph mask."""
    h = glyph.shape[0]
    ranked = []
    for d in digits:
        templ = [render_digit(d, s) for s in sizes]
        templ = [t for t in templ if abs(t.shape[0] - h) <= 3] or templ
        ranked.append((max(ncc(glyph, t) for t in templ), d))
    ranked.sort(reverse=True)
    best, second = ranked[0], ranked[1]
    return best[1], round(best[0], 4), round(best[0] - second[0], 4)


def read_number(glyphs: list[np.ndarray], sizes=QUESTION_SIZES):
    """Classify an ordered list of glyph masks -> (value, min_score, min_margin, digits).

    A leading glyph that is much wider than tall is the minus sign, so signed levels
    ("有符号二进制的 -5 如何表示?") read as a negative value.
    """
    digits, scores, margins = [], [], []
    negative = False
    for i, g in enumerate(glyphs):
        gh, gw = g.shape
        if i == 0 and gh <= 18 and gw / float(max(gh, 1)) >= 1.7:
            negative = True
            continue
        d, s, m = classify_glyph(g, sizes=sizes)
        digits.append(d); scores.append(s); margins.append(m)
    if not digits:
        return None, 0.0, 0.0, []
    value = int("".join(str(d) for d in digits))
    return (-value if negative else value), min(scores), min(margins), digits


# --------------------------------------------------------------------------- frame analysis
def chips_valid(chips: list[dict], w: int, h: int) -> bool:
    """The 8 bit chips are a rigid grid: same row, ~91 px apart, at ~0.85 h.

    Requiring this exact structure is what stops the bot from mistaking some other
    screen (level tree, menus) for the minigame.
    """
    if len(chips) != 8:
        return False
    ys = [c["y"] for c in chips]
    if max(ys) - min(ys) > 4:
        return False
    if not (0.80 * h <= sum(ys) / 8.0 <= 0.91 * h):
        return False
    xs = sorted(c["x"] for c in chips)
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    if max(gaps) - min(gaps) > 5:
        return False
    if not (85 <= gaps[0] <= 97):
        return False
    return True


def is_action_button(b: dict, w: int) -> bool:
    """The 开始 / 提交 buttons are solid rounded rectangles centred on the window."""
    if not (40 <= b["w"] <= 65 and 32 <= b["h"] <= 55):
        return False
    if b["area"] / float(b["w"] * b["h"]) < 0.75:
        return False
    if abs(b["cx"] - w / 2.0) > 35:
        return False
    return True


def dialog_present(img: np.ndarray):
    """Modal dialogs: small red close button top-right, or a wide blue title bar."""
    h, w = img.shape[:2]
    for b in blobs(red_mask(img), 120):
        if (0.08 * h < b["cy"] < 0.36 * h and 0.66 * w < b["cx"] < 0.88 * w
                and 12 <= b["w"] <= 42 and 12 <= b["h"] <= 42):
            return [b["x"], b["y"]]
    for b in blobs(blue_mask(img), 3000, roi=(0, 0, w, int(0.6 * h))):
        if b["w"] >= 300 and 18 <= b["h"] <= 70:
            return [b["x"], b["y"]]
    return None


def find_continue_button(img: np.ndarray):
    """The dialogs' 继续 button is a small blue rounded rectangle in the lower half."""
    h, w = img.shape[:2]
    for b in blobs(blue_mask(img), 400, roi=(0, int(0.35 * h), w, h)):
        if 80 <= b["w"] <= 220 and 22 <= b["h"] <= 55:
            return [round(b["cx"]), round(b["cy"])]
    return None


def _band(h_frac_lo, h_frac_hi, h):
    return int(h * h_frac_lo), int(h * h_frac_hi)


def analyze(img: np.ndarray) -> dict:
    """Full state reading of one frame. All positions are frame-space."""
    h, w = img.shape[:2]
    om, pm, gm, rm = orange_mask(img), pink_mask(img), green_mask(img), red_mask(img)

    out = dict(frame=[w, h], state="unknown")

    # ---- bit chips (row near the bottom) ----
    cy0, cy1 = _band(0.78, 0.92, h)
    chip_blobs = [b for b in blobs(cv2.bitwise_or(pm, gm), 800, roi=(0, cy0, w, cy1))
                  if 40 <= b["w"] <= 90 and 40 <= b["h"] <= 90]
    chips = []
    for i, b in enumerate(sorted(chip_blobs, key=lambda b: b["cx"])):
        on = float((gm[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]] > 0).mean()) > 0.25
        chips.append(dict(idx=i, bit=BITS[i] if i < len(BITS) else None,
                          x=round(b["cx"], 1), y=round(b["cy"], 1), on=on))
    out["chips"] = chips
    out["chips_ok"] = chips_valid(chips, w, h)
    last_chip_x = max((c["x"] for c in chips), default=0)

    # ---- every orange component, classified by geometry + position ----
    cand = blobs(om, 80)
    buttons = [b for b in cand if is_action_button(b, w)]
    # digits plus an optional leading minus sign ("有符号二进制的 -5 如何表示?")
    glyphs_all = [b for b in cand if b not in buttons and 8 <= b["w"] <= 40 and 4 <= b["h"] <= 46
                  and 40 <= b["area"] <= 900]

    qy0, qy1 = _band(0.33, 0.52, h)
    question_glyphs = [b for b in glyphs_all if qy0 <= b["cy"] <= qy1] if out["chips_ok"] else []
    sum_glyphs = ([b for b in glyphs_all if cy0 <= b["cy"] <= cy1 and b["x"] > last_chip_x + 15]
                  if out["chips_ok"] else [])

    bottom_btns = [b for b in buttons if b["cy"] >= _band(0.86, 1.0, h)[0]]
    center_btns = [b for b in buttons if qy0 <= b["cy"] <= _band(0.52, 0.60, h)[1]]
    out["submit_btn"] = [round(bottom_btns[0]["cx"]), round(bottom_btns[0]["cy"])] if bottom_btns else None
    out["start_btn"] = [round(center_btns[0]["cx"]), round(center_btns[0]["cy"])] if center_btns else None
    out["button_blobs"] = [[b["x"], b["y"], b["w"], b["h"], b["area"]] for b in buttons]

    # ---- target digits ----
    out["target"] = None
    if question_glyphs:
        glyphs = [om[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]] for b in question_glyphs]
        val, score, margin, digits = read_number(glyphs, sizes=QUESTION_SIZES)
        out["target_bbox"] = [question_glyphs[0]["x"], question_glyphs[0]["y"],
                              question_glyphs[-1]["x"] + question_glyphs[-1]["w"] - question_glyphs[0]["x"],
                              max(b["h"] for b in question_glyphs)]
        out["target_score"] = score
        out["target_margin"] = margin
        out["target_digits"] = digits
        out["target_glyph_sizes"] = [(g.shape[0], g.shape[1]) for g in glyphs]
        if score >= 0.75 and margin >= 0.09:
            out["target"] = val

    # ---- running sum ----
    out["sum"] = None
    if sum_glyphs:
        sg = [om[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]] for b in sum_glyphs]
        val, score, margin, digits = read_number(sg, sizes=SUM_SIZES)
        out["sum_bbox"] = [sum_glyphs[0]["x"], sum_glyphs[0]["y"],
                           sum_glyphs[-1]["x"] + sum_glyphs[-1]["w"] - sum_glyphs[0]["x"],
                           max(b["h"] for b in sum_glyphs)]
        out["sum_score"] = score
        out["sum_margin"] = margin
        if score >= 0.80 and margin >= 0.10:
            out["sum"] = val

    # ---- white text left of the number = question-template fingerprint ----
    wblobs = blobs(white_mask(img), 40, roi=(0, qy0, w, qy1))
    if wblobs and "target_bbox" in out:
        left = [b for b in wblobs if b["x"] + b["w"] <= out["target_bbox"][0] + 2]
        out["left_text_width"] = (left[-1]["x"] + left[-1]["w"] - left[0]["x"]) if left else 0
        out["left_text_x0"] = left[0]["x"] if left else None
    out["red_text"] = bool(blobs(rm, 200, roi=(0, _band(0.30, 0.62, h)[0], w, _band(0.30, 0.62, h)[1])))

    # ---- state: only a structurally confirmed minigame screen is actionable ----
    if not out["chips_ok"]:
        out["state"] = "unknown"
    elif out["submit_btn"]:
        out["state"] = "question"
    elif out["start_btn"]:
        out["state"] = "ready"
    else:
        out["state"] = "unknown"
    return out


def verify_sum(img: np.ndarray, expected: int, info=None) -> dict:
    """Independently confirm the running sum equals *expected* (render-and-compare)."""
    info = info if info is not None else analyze(img)
    res = dict(expected=expected, read=info.get("sum"), read_score=info.get("sum_score"),
               read_margin=info.get("sum_margin"), render_score=None, chips=[c["on"] for c in info.get("chips", [])])
    if "sum_bbox" in info:
        x, y, w, h = info["sum_bbox"]
        glyphs = orange_mask(img)[y:y + h, x:x + w]
        best = 0.0
        for size in SUM_SIZES:
            t = render_text(str(expected), size)
            if abs(t.shape[0] - h) <= 3:
                best = max(best, ncc(glyphs, t))
        res["render_score"] = round(best, 4)
    return res


def bits_of(value: int) -> list[int]:
    return [b for b in BITS if value & b]
