# Turing Complete · Binary Racer / Negative Numbers / Hexadecimal Speedrun Auto-Solver

[中文](README.md) | **English**

> Automates Turing Complete Binary Racer & Negative-number minigames via screen recognition + real mouse input

The bot reads the question number, computes which bits to toggle, clicks the bit chips,
**verifies the running sum**, presses *Submit*, and records every question together with
its index, elapsed time and result. Verified full clear with zero wrong answers:
**63/63** on the unsigned level and **53/53** on the signed (negative) level.

## Demo

Live run covering both the Binary Racer and Negative-number levels:
**[Watch on Bilibili](https://www.bilibili.com/video/BV1H1Hp64Ese)**

## Results

| Level | Question style | Score |
| --- | --- | --- |
| Binary Racer | `二进制下，37 如何表示?` | **63/63**, level 7 (max) |
| Negative numbers | `有符号二进制的 -5 如何表示?` | **53/53**, level 6 (max) |
| Hexadecimal speedrun | `十六进制的 0x1F 如何表示?` | **63/63**, level 7 (max) |

## How it works

*Turing Complete* uses its own GLFW/OpenGL engine — the accessibility tree is empty and
`PrintWindow` returns pure black — so the bot works on visible pixels plus real input:

| Step | Approach |
| --- | --- |
| Capture | DPI-aware screen grab of the **game window rectangle only**, scaled into a fixed "frame space" (1568×980). Works even when the window is not fullscreen. |
| Recognition | Candidate glyphs are rendered offline with the game font and matched against the on-screen orange glyphs by normalised cross-correlation (NCC). Real questions score 0.86–0.99. Three notations are auto-detected: **decimal**, **signed decimal** and **hexadecimal (`0x1F`)**. A leading wide-and-short glyph is a minus sign; a `0` followed by a short squarish glyph is the `0x` prefix. |
| Locating | The 8 bit chips are validated as a rigid grid (equal spacing, single row). Any other screen becomes `unknown` and is **never clicked blindly**. |
| Solving | Target `v` → toggle the bits where `v & bit`. Python's `&` is two's complement for negative values, so the unsigned and signed levels share the same line of code. |
| Verification | Before submitting: (1) all 8 chip states must match the target, and (2) if the game shows `= sum`, the reading must equal the target. Higher levels hide the sum display, in which case the chip states are authoritative. |
| State machine | ready / question / dialog / unknown. Dialogs are found by their red close button or wide blue title bar and dismissed by clicking 继续. |

## Requirements

- Windows 10/11, Python 3.9+ (tested on 3.11)
- `pip install pillow numpy opencv-python`
- *Turing Complete* installed via Steam; borderless fullscreen recommended (tested at 2560×1600, other resolutions adapt)
- **Font**: the bot auto-detects the game's `asset/font/NoroshiCode_Bold.ttf` from your Steam libraries.
  You can also point at it with the `TC_BINRACER_FONT` environment variable or `--font <path>`.
  The repository contains **no game assets**.

## Quick start

```bat
pip install pillow numpy opencv-python
:: open the game, enter the level and stop at the "准备好了吗? / 开始" screen, then run:
run_standalone.bat
```

Do not touch the mouse while it runs. The console minimises itself to the taskbar
(hover the icon for live progress). Press **F12** to abort at any time.

## Command line

| Option | Meaning |
| --- | --- |
| `--input {sendinput,file}` | `sendinput` injects real mouse input from this process (default, recommended); `file` delegates clicks to an external agent |
| `--capture {screen,file}` | Frame source; combined with `--input=file` this allows capturing an occluded window |
| `--ipc-dir DIR` | Request/ack directory for `--input=file` |
| `--max-seconds N` | Run time limit (default 480) |
| `--lost-seconds N` | How long the game may stay unrecognisable before stopping safely (default 8) |
| `--save-shots` | Save per-question crops |
| `--font PATH` | Path to the game font |
| `--log-file PATH` | Also write output to a file |
| `--no-console-min` | Keep the console visible |

## Recorded output

One directory per run, `logs/run_<timestamp>/`:

| Field | Meaning |
| --- | --- |
| `q_index` | Question index within the run |
| `target` | Recognised question number (may be negative) |
| `bits` | Toggled bits, e.g. `128 + 64 + 16 + 1` |
| `submitted_sum` | Running-sum reading at submit time ("-" when the game hides it) |
| `verify_ok` | Whether pre-submit verification passed |
| `target_score` / `target_margin` | Recognition confidence / gap to the runner-up |
| `elapsed_ms` | Time spent on this question |
| `clicks` | Clicks issued for this question |
| `result` | accepted / pending / unknown … |

Also `questions.jsonl`, per-question crops in `shots/`, `summary.md` and `DONE.json` (stop reason).

## Robustness notes (bugs that actually bit us)

| Symptom | Cause | Fix |
| --- | --- | --- |
| Repeated clicks produced wrong sums | Verifying immediately after clicking read the *old* state, so already-correct chips were "corrected" (i.e. toggled off) | `settle()` waits for a stable chip row before judging; 200 ms per-position debounce |
| The same question was re-solved after a lost submit | A swallowed submit click led to re-answering | `pending_sig` guard: while the identical question with correct chips is on screen the bot only waits, and re-presses Submit at most once |
| A transition frame killed the whole run | A question fading out (plain background) was mistaken for conflicting reads | 2-of-3 majority vote; genuine disagreement now just waits for the game to move on |
| Every click landed in the wrong place | The code assumed "fullscreen at (0,0)" | Capture and click mapping now follow the **real window rectangle** |
| Coordinates off by a factor of 1.5 | The process started DPI-unaware, which cannot be changed afterwards | DPI awareness is set at module import |
| The game was knocked out of fullscreen (taskbar stuck on top) | An early version called `ShowWindow(SW_RESTORE)` on the borderless window | Removed; the window's show state is never touched, and a warning is printed if it looks abnormal |

## Known limits

- Clearing a minigame (level 7 for Binary Racer, level 6 for Negative numbers) ends with a
  summary and returns to the level tree; re-enter the level manually to play again.
- Higher levels hide the `= sum` display; verification then relies on chip states (still exact).
- If the game is occluded and the external capture agent is not in use, the bot stops safely instead of guessing.

## License

[MIT](LICENSE). The repository ships no game assets; the font is read from your own game
installation at runtime.

## Disclaimer

- For personal automation practice on a single-player game; check that your usage complies
  with the game's EULA.
- *Turing Complete* and its assets belong to their respective owners. This project is not
  affiliated with them.
- `logs/` (which may contain screenshots of your desktop) is git-ignored and never uploaded.
