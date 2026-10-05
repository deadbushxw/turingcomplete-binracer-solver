# -*- coding: utf-8 -*-
"""Capture and input backends for the Binary Racer bot.

Two input paths exist because of how this machine is set up:

* kind="file"      - the bot runs inside a sandbox that silently swallows SendInput.
                     Click requests are written to an IPC directory and performed by a
                     small driver-side proxy (cua-driver MCP click).
* kind="sendinput" - the bot runs as a normal user process (double-click a .bat),
                     so it can inject real mouse input directly.

Capture always uses ImageGrab: it is ~48 ms for the whole screen and works as long
as the game window is foreground (which the driver's foreground clicks guarantee).
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as w
import json
import os
import time

import numpy as np
from PIL import ImageGrab

from . import vision

# MODULE-LEVEL DPI awareness: it must be set before any window/geometry query and
# can never be changed afterwards. A DPI-unaware process sees the game as 1707x1067
# instead of 2560x1600, which would shift every click by a factor of 1.5.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

FRAME_SIZE = (vision.FRAME_W, vision.FRAME_H)


# --------------------------------------------------------------------------- capture
def game_rect():
    """(left, top, width, height) of the game window, or None if not found.

    Never assume the window is fullscreen at (0,0): after any window-state change
    it can be windowed, and then both the capture crop and the click mapping must
    follow the real rectangle or every click lands in the wrong place.
    """
    try:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass
        u = ctypes.windll.user32
        hwnd = u.FindWindowW(None, "Turing Complete")
        if not hwnd:
            return None

        class _RECT(ctypes.Structure):
            _fields_ = (("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long))
        r = _RECT()
        u.GetWindowRect(hwnd, ctypes.byref(r))
        w, h = r.right - r.left, r.bottom - r.top
        if w < 200 or h < 200:
            return None
        return (r.left, r.top, w, h)
    except Exception:
        return None


def frame_to_physical(fx: float, fy: float):
    """Frame-space (1568x980) -> absolute screen pixels, following the real window."""
    rect = game_rect()
    if rect:
        return (int(round(rect[0] + fx * rect[2] / float(vision.FRAME_W))),
                int(round(rect[1] + fy * rect[3] / float(vision.FRAME_H))))
    return vision.to_physical(fx, fy)


class ScreenCapture:
    """Grab the primary screen and return a frame-space (1568x980) BGR array."""

    name = "screen"

    def __init__(self):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass

    def grab(self) -> np.ndarray:
        import cv2
        rect = game_rect()
        if rect:                                     # crop the game window itself
            img = ImageGrab.grab(bbox=(rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]),
                                 all_screens=True)
        else:
            img = ImageGrab.grab(all_screens=True)
        a = np.array(img)[:, :, ::-1].copy()          # RGB -> BGR
        a = a[:vision.SCREEN_H, :vision.SCREEN_W]
        return np.ascontiguousarray(cv2.resize(a, FRAME_SIZE, interpolation=cv2.INTER_AREA))


class FileCapture:
    """Fallback: read the newest frame written by the driver-side proxy."""

    name = "file"

    def __init__(self, ipc_dir: str):
        self.path = os.path.join(ipc_dir, "frame.png")

    def grab(self) -> np.ndarray:
        import cv2
        img = cv2.imread(self.path)
        if img is None:
            raise RuntimeError("no frame available at %s" % self.path)
        return img


# --------------------------------------------------------------------------- input
class _MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx", w.LONG), ("dy", w.LONG), ("mouseData", w.DWORD), ("dwFlags", w.DWORD),
                ("time", w.DWORD), ("dwExtraInfo", ctypes.POINTER(w.ULONG)))


class _INPUTUNION(ctypes.Union):
    _fields_ = (("mi", _MOUSEINPUT),)


class _INPUT(ctypes.Structure):
    _fields_ = (("type", w.DWORD), ("u", _INPUTUNION))


class SendInputBackend:
    """Real mouse input, for a normal (non-sandboxed) process."""

    name = "sendinput"
    MOVE, ABSOLUTE, LEFTDOWN, LEFTUP = 0x0001, 0x8000, 0x0002, 0x0004

    def __init__(self):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass
        self._u32 = ctypes.windll.user32
        self._sw = (self._u32.GetSystemMetrics(0) - 1) or 1
        self._sh = (self._u32.GetSystemMetrics(1) - 1) or 1

    def click(self, fx: float, fy: float) -> None:
        px, py = frame_to_physical(fx, fy)
        nx = int(px * 65535 / self._sw)
        ny = int(py * 65535 / self._sh)
        for flags, dx, dy in ((self.MOVE | self.ABSOLUTE, nx, ny), (self.LEFTDOWN, 0, 0), (self.LEFTUP, 0, 0)):
            inp = _INPUT(type=0, u=_INPUTUNION(mi=_MOUSEINPUT(dx, dy, 0, flags, 0, None)))
            self._u32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
            time.sleep(0.012)

    def focus(self) -> None:
        """Put the game window in the foreground WITHOUT touching its show state.

        NOTE: never call ShowWindow()/SW_RESTORE here. The game is a borderless
        fullscreen window; restoring it drops it out of fullscreen and the Windows
        taskbar then stays on top of the game until it is restarted.
        """
        try:
            u = ctypes.windll.user32
            hwnd = u.FindWindowW(None, "Turing Complete")
            if not hwnd or u.GetForegroundWindow() == hwnd:
                return
            fg = u.GetForegroundWindow()
            ft = u.GetWindowThreadProcessId(fg, None)
            tt = u.GetWindowThreadProcessId(hwnd, None)
            u.AttachThreadInput(ft, tt, True)
            u.SetForegroundWindow(hwnd)
            u.AttachThreadInput(ft, tt, False)
        except Exception:
            pass


class FileInputBackend:
    """Click by handing a request to the driver-side proxy through files."""

    name = "file"

    def __init__(self, ipc_dir: str, timeout: float = 25.0):
        self.dir = ipc_dir
        self.timeout = timeout
        os.makedirs(ipc_dir, exist_ok=True)
        self.req = os.path.join(ipc_dir, "req.json")
        self.ack = os.path.join(ipc_dir, "ack.json")
        self.seq = 0

    def _send(self, action: str, **kw) -> dict:
        self.seq += 1
        # Tokens are unique per run (pid + counter + ns), so a leftover request or
        # ack from a previous run can never be mistaken for this one.
        token = "%d-%d-%d" % (os.getpid(), self.seq, time.time_ns())
        try:
            os.remove(self.ack)
        except OSError:
            pass
        payload = dict(id=token, action=action, ts=time.time(), **kw)
        tmp = self.req + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        for _try in range(60):                      # Windows may deny the replace while the proxy reads
            try:
                os.replace(tmp, self.req)
                break
            except PermissionError:
                time.sleep(0.005)
        else:
            with open(self.req, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            try:
                os.remove(tmp)
            except OSError:
                pass
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            try:
                with open(self.ack, "r", encoding="utf-8") as fh:
                    ack = json.load(fh)
                if ack.get("id") == token:
                    return ack
            except (OSError, ValueError):
                pass
            time.sleep(0.008)
        raise TimeoutError("no ack for %s request %s" % (action, token))

    def click(self, fx: float, fy: float) -> None:
        self._send("click", x=round(float(fx), 1), y=round(float(fy), 1))

    def focus(self) -> None:
        self._send("focus")

    def request_frame(self) -> None:
        """Ask the proxy to fetch the game window content (works even when occluded)."""
        self._send("capture")


def make_input(kind: str, ipc_dir):
    if kind == "sendinput":
        return SendInputBackend()
    if kind == "file":
        if not ipc_dir:
            raise SystemExit("--ipc-dir is required for --input=file")
        return FileInputBackend(ipc_dir)
    raise SystemExit("unknown input backend: %s" % kind)


def make_capture(kind: str, ipc_dir):
    if kind == "screen":
        return ScreenCapture()
    if kind == "file":
        if not ipc_dir:
            raise SystemExit("--ipc-dir is required for --capture=file")
        return FileCapture(ipc_dir)
    raise SystemExit("unknown capture backend: %s" % kind)


def abort_pressed() -> bool:
    """F12 aborts immediately (the bot owns the mouse while it runs)."""
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x7B) & 0x8000)
    except Exception:
        return False


def read_frame_file(ipc_dir: str):
    """Read the window frame the proxy saved (frame space, same 1568x980 layout)."""
    import cv2
    img = cv2.imread(os.path.join(ipc_dir, "frame.png"))
    if img is None:
        raise RuntimeError("proxy did not provide a frame")
    return img


def console_window():
    try:
        return ctypes.windll.kernel32.GetConsoleWindow()
    except Exception:
        return 0


def minimize_console() -> None:
    """Keep the console out of the way of the fullscreen game (it stays in the taskbar)."""
    try:
        h = console_window()
        if h:
            ctypes.windll.user32.ShowWindow(h, 6)      # SW_MINIMIZE
    except Exception:
        pass


def set_console_title(text: str) -> None:
    try:
        ctypes.windll.kernel32.SetConsoleTitleW(text)
    except Exception:
        pass
