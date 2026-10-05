# -*- coding: utf-8 -*-
"""Automated solver for the Turing Complete "Binary Racer" (二进制速算) level.

Loop: read the question number -> toggle bit chips -> verify the running sum ->
submit -> record question / index / elapsed time / submitted answer.

The level passes at level 3 but keeps counting, so the bot answers until the game
ends. Per the agreed policy it stops at the first failure and writes a report.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from binracer import backend as bk
from binracer import vision


class Bot:
    def __init__(self, args):
        self.args = args
        self.cap = bk.make_capture(args.capture, args.ipc_dir)
        self.inp = bk.make_input(args.input, args.ipc_dir)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(args.log_dir, "run_" + self.run_id)
        self.shot_dir = os.path.join(self.log_dir, "shots")
        os.makedirs(self.shot_dir, exist_ok=True)
        self.jsonl = open(os.path.join(self.log_dir, "questions.jsonl"), "a", encoding="utf-8")
        self.csv_fh = open(os.path.join(self.log_dir, "questions.csv"), "a", newline="", encoding="utf-8-sig")
        self.csv = csv.writer(self.csv_fh)
        if self.csv_fh.tell() == 0:
            self.csv.writerow(["run_id", "q_index", "target", "bits", "submitted_sum", "verify_ok",
                               "render_score", "target_score", "target_margin", "elapsed_ms",
                               "clicks", "result", "shot"])
        self.q_index = 0
        self.finished = False
        self.started = False
        self.records = []
        self.n_clicks = 0
        self.unknown_streak = 0
        self.ready_streak = 0
        self.pending_sig = None       # target we already answered (waiting for the game)
        self.pending_since = 0.0
        self.resubmitted = False
        self.last_click_at = {}
        self.started_at = time.time()

    # ------------------------------------------------------------------ plumbing
    def read(self):
        frame = self.cap.grab()
        info = vision.analyze(frame)
        if not info["chips_ok"] and isinstance(self.inp, bk.FileInputBackend):
            # The game window may be covered by another window: ask the driver-side proxy
            # for the window content, which it can capture even while occluded.
            try:
                self.inp.request_frame()
                if isinstance(self.cap, bk.ScreenCapture):
                    frame2 = bk.read_frame_file(self.args.ipc_dir)
                else:
                    frame2 = self.cap.grab()
                info2 = vision.analyze(frame2)
                if info2["chips_ok"]:
                    return frame2, info2
            except Exception:
                pass
        return frame, info

    def wait_for_proxy(self, timeout=90.0):
        """Wait until the driver-side proxy is polling, so no click is ever lost."""
        if not isinstance(self.inp, bk.FileInputBackend):
            return
        flag = os.path.join(self.args.ipc_dir, "proxy_ready.json")
        t0 = time.time()
        while time.time() - t0 < timeout:
            if os.path.exists(flag):
                return
            time.sleep(0.1)
        raise RuntimeError("driver proxy never became ready (%s)" % flag)

    def ensure_game_visible(self, tries=10):
        frame, info = None, None
        for _ in range(tries):
            frame, info = self.read()
            if info["chips_ok"]:
                return frame, info
            self.inp.focus()
            time.sleep(0.4)
        return frame, info

    def save_shot(self, frame, tag, full=False):
        if not self.args.save_shots:
            return None
        if full:
            path = os.path.join(self.shot_dir, tag + ".png")
            cv2.imwrite(path, frame)
            return os.path.relpath(path, self.log_dir)
        q = frame[320:500, :]
        c = frame[775:975, :]
        cv2.imwrite(os.path.join(self.shot_dir, tag + "_q.png"), q)
        cv2.imwrite(os.path.join(self.shot_dir, tag + "_c.png"), c)
        return tag

    # ------------------------------------------------------------------ per question
    def chips_match(self, info, want):
        return bool(info.get("chips_ok")) and all(int(c["on"]) == bit for bit, c in zip(want, info["chips"]))

    def settle(self, want=None, timeout=0.35):
        """Re-read until the chip row stops changing.

        The game renders a click over a few frames. Verifying immediately after
        clicking used to read the *old* state and then "correct" chips that were
        already right - clicking them twice and producing a wrong sum.
        """
        t0 = time.time()
        frame, info = self.read()
        last = tuple(int(c["on"]) for c in info["chips"]) if info.get("chips_ok") else None
        while time.time() - t0 < timeout:
            if want is not None and self.chips_match(info, want):
                break                       # already exactly right - no extra read needed
            time.sleep(0.025)
            frame2, info2 = self.read()
            sig = tuple(int(c["on"]) for c in info2["chips"]) if info2.get("chips_ok") else None
            frame, info = frame2, info2
            if sig is not None and sig == last:
                break
            last = sig
        return frame, info

    def click(self, pos):
        """Click with a per-position debounce so a chip can never be toggled twice
        by chattering control flow."""
        key = (round(pos[0] / 6.0), round(pos[1] / 6.0))
        now = time.time()
        last = self.last_click_at.get(key, 0.0)
        if now - last < 0.12:
            time.sleep(0.12 - (now - last))
        self.last_click_at[key] = time.time()
        self.inp.click(pos[0], pos[1])
        self.n_clicks += 1

    def solve(self, frame, info):
        q_t0 = time.time()
        target = info["target"]

        # dialog drawn over the question (level complete / summary): keep playing
        if target is None and vision.dialog_present(frame):
            pos = vision.find_continue_button(frame)
            if pos is not None:
                self.click(pos)
                time.sleep(0.6)
                return True
        tries = 0
        while target is None and tries < 8:
            if vision.dialog_present(frame):
                pos = vision.find_continue_button(frame)
                if pos is not None:
                    self.click(pos)
                    time.sleep(0.6)
                    return True
            time.sleep(0.12)
            frame, info = self.read()
            target = info["target"]
            tries += 1
        if target is None:
            if vision.dialog_present(frame):
                pos = vision.find_continue_button(frame)
                if pos is not None:
                    self.click(pos)
                    time.sleep(0.6)
                    return True
            # Do not kill the run here: the number may simply be mid-animation or the
            # question may already be over. Wait for the game to move on by itself.
            self.save_shot(frame, "unreadable", full=True)
            self.anomalies = getattr(self, "anomalies", 0) + 1
            print("[--] 未读到题目,等待游戏状态变化 (#%d)" % self.anomalies, flush=True)
            time.sleep(0.25)
            return True

        # marginal read: require a second independent frame to agree
        if info.get("target_score", 0) < 0.85 or info.get("target_margin", 0) < 0.20:
            time.sleep(0.18)
            frame2, info2 = self.read()
            if info2.get("target") != target:
                # reads disagree: take a third one and use a 2-of-3 majority. A single
                # odd frame (question fading out, transition) must never end the run.
                time.sleep(0.18)
                frame3, info3 = self.read()
                votes: dict = {}
                for v in (info.get("target"), info2.get("target"), info3.get("target")):
                    if v is not None:
                        votes[v] = votes.get(v, 0) + 1
                winner, best = None, 0
                for v, c in votes.items():
                    if c > best:
                        winner, best = v, c
                if best < 2:
                    self.save_shot(frame3, "ambiguous", full=True)
                    self.anomalies = getattr(self, "anomalies", 0) + 1
                    print("[--] 三次读数都不一致,放弃本题(等游戏自己走)" , flush=True)
                    time.sleep(0.25)
                    return True
                target = winner
                if info3.get("target") == winner:
                    frame, info = frame3, info3
                elif info2.get("target") == winner:
                    frame, info = frame2, info2
                print("[--] 读数抖动,按多数取 %s" % winner, flush=True)
            else:
                frame, info = frame2, info2

        want = [1 if target & b else 0 for b in vision.BITS]

        # If the screen is still exactly what we answered a moment ago, the game has
        # not processed our submit yet: wait instead of clicking the same bits again.
        if (self.pending_sig == target and self.chips_match(info, want)
                and time.time() - self.pending_since < 2.0):
            time.sleep(0.15)
            return True

        clicks = 0
        for _attempt in range(4):
            if not info.get("chips_ok"):
                frame, info = self.read()
                time.sleep(0.1)
                continue
            todo = [c for bit, c in zip(want, info["chips"]) if int(c["on"]) != bit]
            if not todo:
                break
            for c in todo:
                self.click((c["x"], c["y"]))
                clicks += 1
            frame, info = self.settle(want)                   # let the game render the clicks
            if self.chips_match(info, want):
                break

        verify = vision.verify_sum(frame, target, info)
        state_ok = self.chips_match(info, want)
        sum_read = verify.get("read")
        ok = state_ok and (sum_read is None or sum_read == target)

        # one correction round - only for chips still wrong on a *settled* frame
        if clicks and not state_ok and info.get("chips_ok"):
            for c, bit in zip(info["chips"], want):
                if int(c["on"]) != bit:
                    self.click((c["x"], c["y"]))
                    clicks += 1
            frame, info = self.settle()
            verify = vision.verify_sum(frame, target, info)
            state_ok = self.chips_match(info, want)
            sum_read = verify.get("read")
            ok = state_ok and (sum_read is None or sum_read == target)

        self.save_shot(frame, "q%03d" % self.q_index)
        self.pending_sig = target
        self.pending_since = time.time()
        self.resubmitted = False
        if info.get("submit_btn"):
            self.click(info["submit_btn"])
        t_submit = time.time()
        result = self.await_result(target, frame)
        if result == "unknown":
            time.sleep(0.4)
            f2, i2 = self.read()
            if i2["state"] == "question" and i2.get("target") is not None:
                result = "assumed_accepted" if i2["target"] != target or not self.chips_match(i2, want) else "unknown"
        if result == "unknown":
            # our submit may have been swallowed: press 提交 once more, never more
            f3, i3 = self.read()
            if i3["state"] == "question" and self.chips_match(i3, want) and i3.get("submit_btn") and not self.resubmitted:
                self.resubmitted = True
                self.click(i3["submit_btn"])
                self.pending_since = time.time()
                time.sleep(0.4)
                f4, i4 = self.read()
                if i4["state"] == "question" and (i4.get("target") != target or not self.chips_match(i4, want)):
                    result = "accepted"
        if result == "unknown":
            # Still the same live question with our chips correct: the game has not
            # confirmed yet, but it is clearly NOT over - keep playing instead of
            # stopping early (the pending guard prevents any re-clicking here).
            f5, i5 = self.read()
            if i5["state"] == "question" and self.chips_match(i5, want):
                result = "pending"
            else:
                self.save_shot(f5, "lost_after_submit", full=True)
                self.finish("game_over", f5, None)
                return False

        self.q_index += 1
        rec = dict(run_id=self.run_id, q_index=self.q_index, target=target,
                   bits=vision.bits_of(target), submitted_sum=sum_read,
                   verify_ok=bool(ok), sum_visible=sum_read is not None,
                   render_score=verify.get("render_score"),
                   target_score=info.get("target_score"), target_margin=info.get("target_margin"),
                   elapsed_ms=round((t_submit - q_t0) * 1000, 1), clicks=clicks, result=result,
                   t_iso=datetime.now().isoformat(timespec="milliseconds"))
        self.records.append(rec)
        self.jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.jsonl.flush()
        self.csv.writerow([rec["run_id"], rec["q_index"], rec["target"], " + ".join(map(str, rec["bits"])),
                           rec["submitted_sum"], rec["verify_ok"], rec["render_score"], rec["target_score"],
                           rec["target_margin"], rec["elapsed_ms"], rec["clicks"], rec["result"], ""])
        self.csv_fh.flush()
        try:
            bk.set_console_title("二进制速算机器人 | 已完成 %d 题 | 最近: %s -> %s (%.0fms)" % (
                self.q_index, target, result, rec["elapsed_ms"]))
        except Exception:
            pass
        print("[%3d] target=%-4s bits=%-18s sum=%-5s verify=%-5s %5.0fms %s" % (
            rec["q_index"], rec["target"], ",".join(map(str, rec["bits"])), rec["submitted_sum"],
            rec["verify_ok"], rec["elapsed_ms"], rec["result"]), flush=True)
        return result in ("accepted", "assumed_accepted", "pending")

    def await_result(self, target, prev_frame, timeout=1.2):
        """Did the game move on (accepted) or drop to the ready screen (run over)?

        Compares tightly cropped regions: the question number and the chip row. A
        changed digit covers only ~1% of the whole line band, so a coarse whole-band
        average would miss it (this cost us the target=0 case).
        """
        chip_roi = (slice(790, 880), slice(0, vision.FRAME_W))
        q_roi = (slice(395, 448), slice(690, 900))
        prev_chip = prev_frame[chip_roi].astype(np.int16)
        prev_q = prev_frame[q_roi].astype(np.int16)
        t0 = time.time()
        while time.time() - t0 < timeout:
            frame = self.cap.grab()
            chip_changed = float((np.abs(frame[chip_roi].astype(np.int16) - prev_chip) > 50).mean())
            q_changed = float((np.abs(frame[q_roi].astype(np.int16) - prev_q) > 50).mean())
            if chip_changed > 0.005 or q_changed > 0.01:
                info = vision.analyze(frame)
                if info["state"] == "ready":
                    self.save_shot(frame, "failed", full=True)
                    return "rejected"
                if info["state"] == "question":
                    return "accepted"
            time.sleep(0.03)
        return "unknown"

    # ------------------------------------------------------------------ run
    def run(self):
        self.wait_for_proxy()
        print("run_id=%s input=%s capture=%s" % (self.run_id, self.inp.name, self.cap.name), flush=True)
        try:
            u = ctypes.windll.user32
            hwnd = u.FindWindowW(None, "Turing Complete")
            if hwnd and u.IsZoomed(hwnd):
                print("[!] 游戏窗口处于'最大化/还原'状态(任务栏可能盖住游戏底部,影响提交按钮),"
                      "建议重启游戏或切回全屏", flush=True)
        except Exception:
            pass
        rect = bk.game_rect()
        if rect:
            print("游戏窗口: %s,屏幕 %dx%d%s" % (rect, vision.SCREEN_W, vision.SCREEN_H,
                  "" if abs(rect[2] - vision.SCREEN_W) < 8 and abs(rect[3] - vision.SCREEN_H) < 8
                  else "  [!] 非全屏,已按窗口实际位置换算"), flush=True)
        frame, info = self.ensure_game_visible()
        deadline = self.started_at + self.args.max_seconds
        start_clicked_at = None
        start_retries = 0
        while time.time() < deadline:
            if bk.abort_pressed():
                self.finish("aborted_by_user", frame, None)
                return
            state = info["state"]
            if state == "question":
                self.started = True
                self.unknown_streak = 0
                self.ready_streak = 0
                if not self.solve(frame, info):
                    self.finish("game_over", frame, None)
                    return
            elif state == "ready":
                self.unknown_streak = 0
                if self.q_index > 0:
                    self.ready_streak += 1
                    if self.ready_streak >= 3:          # persistent, not a transition frame
                        self.finish("game_over", frame, None)
                        return
                    time.sleep(0.15)
                if start_clicked_at is None or time.time() - start_clicked_at > 5.0:
                    if start_retries >= 3:
                        self.finish("start_failed", frame, None)
                        return
                    start_retries += 1
                    start_clicked_at = time.time()
                    if info.get("start_btn"):
                        self.click(info["start_btn"])
                        self.started = True
                    time.sleep(self.args.start_settle)
                else:
                    time.sleep(0.15)
            else:
                # Not recognisably the minigame (another window on top, or a menu).
                # Never click blindly; try to bring the game forward and wait longer.
                self.ready_streak = 0
                self.unknown_streak += 1
                if self.unknown_streak % 10 == 0:
                    self.inp.focus()
                if self.unknown_streak >= int(self.args.lost_seconds / 0.15):
                    self.finish("lost_screen", frame, None)
                    return
                time.sleep(0.15)
            frame, info = self.read()
        self.finish("time_limit", frame, None)

    def finish(self, reason, frame=None, extra=None):
        if self.finished:
            return
        self.finished = True
        n = len(self.records)
        good = sum(1 for r in self.records if r["result"] in ("accepted", "assumed_accepted"))
        el = [r["elapsed_ms"] for r in self.records] or [0.0]
        lines = ["# Binary Racer run %s" % self.run_id, "",
                 "- 结束原因: **%s**" % reason,
                 "- 完成题数: **%d**(被接受 %d 题)" % (n, good),
                 "- 游戏内级数: 约 第 %d 级(每 9 题升 1 级)" % ((n + 8) // 9),
                 "- 平均用时: %.0f ms,最慢 %.0f ms" % (sum(el) / len(el), max(el)),
                 "- 点击次数: %d" % self.n_clicks,
                 "- 输入后端: %s,截图后端: %s" % (self.inp.name, self.cap.name), ""]
        if extra:
            lines += ["    " + json.dumps(extra, ensure_ascii=False), ""]
        with open(os.path.join(self.log_dir, "summary.md"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        if frame is not None:
            try:
                cv2.imwrite(os.path.join(self.log_dir, "final_%s.png" % reason), frame)
            except Exception:
                pass
        print("FINISH %s after %d questions (%d accepted)" % (reason, n, good), flush=True)
        self.jsonl.close()
        self.csv_fh.close()
        with open(os.path.join(self.log_dir, "DONE.json"), "w", encoding="utf-8") as fh:
            json.dump(dict(reason=reason, questions=n, accepted=good, run_id=self.run_id), fh)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="file", choices=["file", "sendinput"])
    ap.add_argument("--capture", default="screen", choices=["screen", "file"])
    ap.add_argument("--ipc-dir", default=None)
    ap.add_argument("--log-dir", default=os.path.join(here, "logs"))
    ap.add_argument("--max-seconds", type=float, default=480)
    ap.add_argument("--start-settle", type=float, default=0.5)
    ap.add_argument("--save-shots", action="store_true")
    ap.add_argument("--lost-seconds", type=float, default=8.0,
                    help="how long the game may stay unrecognisable before giving up")
    ap.add_argument("--log-file", default=None, help="tee stdout/stderr here")
    ap.add_argument("--no-console-min", action="store_true", help="keep the console visible")
    ap.add_argument("--font", default=None, help="path to the game's NoroshiCode_Bold.ttf")
    args = ap.parse_args()
    if args.log_file:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
            fh = open(args.log_file, "a", encoding="utf-8", buffering=1)
            sys.stdout = fh
            sys.stderr = fh
        except OSError:
            pass
    try:
        vision.font_path(args.font)
    except FileNotFoundError as e:
        print(e)
        raise SystemExit(2)
    if not args.no_console_min:
        bk.minimize_console()
        bk.set_console_title("二进制速算机器人 | 启动中…")
    bot = Bot(args)
    try:
        bot.run()
    except BaseException:
        import traceback
        tb = traceback.format_exc()
        try:
            with open(os.path.join(bot.log_dir, "ERROR.txt"), "w", encoding="utf-8") as fh:
                fh.write(tb)
        except OSError:
            pass
        print(tb, flush=True)
        try:
            bot.finish("exception", None, dict(last_line=tb.strip().splitlines()[-1]))
        except Exception:
            pass


if __name__ == "__main__":
    main()
