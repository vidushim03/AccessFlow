"""Verify the shipped HeadTracker (reference-style EMA) settles when the head is still.

Runs the real accessflow.inputs.vision.HeadTracker so the numbers reflect the
code that actually ships, not a copy.
"""
import math
import random

from accessflow.inputs.vision import HeadTracker

SCREEN_W, SCREEN_H = 1920, 1080


def px(a, b):
    """Distance between two normalised points, in screen pixels."""
    return math.hypot((a[0] - b[0]) * SCREEN_W, (a[1] - b[1]) * SCREEN_H)


def main():
    random.seed(7)
    neutral = (0.5, 0.5)
    still = [(neutral[0] + random.gauss(0, 0.004), neutral[1] + random.gauss(0, 0.004))
             for _ in range(200)]

    # 1. Still user with camera jitter, pointer starting off-centre.
    t = HeadTracker()
    anchor = (0.3, 0.4)
    out = []
    for s in still:
        r = t.update(s, anchor=anchor if t.smoothed is None else None)
        if r:
            out.append(r)
    print(f"Still user: anchor={anchor} -> final={tuple(round(v, 4) for v in out[-1])} "
          f"drift={px(out[-1], out[0]):.1f}px over {len(out)} frames, "
          f"max_jitter={max(px(o, out[0]) for o in out):.1f}px")

    # 2. Touchpad takeover mid-session: pointer moves, head stays still.
    t2 = HeadTracker()
    out2 = []
    for s in still:
        r = t2.update(s, anchor=(0.3, 0.4) if t2.smoothed is None else None)
        if r:
            out2.append(r)
    t2.smoothed = None  # what _reset_pointer_state() does on takeover
    for s in still:
        r = t2.update(s, anchor=(0.8, 0.2) if t2.smoothed is None else None)
        if r:
            out2.append(r)
    settle = out2[-20:]
    print(f"Takeover: re-anchored to (0.8, 0.2), final="
          f"{tuple(round(v, 4) for v in out2[-1])}, "
          f"settle drift over last 20 frames={px(settle[-1], settle[0]):.1f}px")

    # 3. Slow head turn should produce smooth motion.
    moving = [(0.5 + 0.0008 * i + random.gauss(0, 0.003), 0.5 + random.gauss(0, 0.003))
              for i in range(200)]
    t3 = HeadTracker()
    out3 = [r for s in moving if (r := t3.update(s, anchor=(0.5, 0.5) if t3.smoothed is None else None))]
    print(f"Slow turn: start={tuple(round(v, 4) for v in out3[0])} -> "
          f"end={tuple(round(v, 4) for v in out3[-1])}, "
          f"moved {px(out3[-1], out3[0]):.1f}px right")


if __name__ == "__main__":
    main()
