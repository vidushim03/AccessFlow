# AccessFlow

Windows desktop MVP for adaptive multimodal assistive interaction. Python 3.11 is recommended.

## Start

    .venv\Scripts\python.exe main.py

Dependencies are listed in requirements.txt.

## Connected inputs

- Head tracking and hand gestures share a single OpenCV camera capture worker. Camera startup tries DirectShow, Media Foundation, and OpenCV's automatic backend on the first two device indices, and requires an actual frame before proceeding. If no camera opens, AccessFlow explains that another app may be using it and points to Windows camera privacy permissions. Face/hand models are fetched from official MediaPipe model storage on first camera start.
- Head tracking moves the pointer from the raw nose-tip position relative to a learned neutral pose, smoothed with a plain 0.18 EMA. The neutral pose is learned from a short stable window while the pointer is held still, so a bad first frame or sitting down mid-detection cannot bias the cursor. The pointer is anchored to wherever it already is, and re-anchors when head control takes over from another input, so starting a session, switching methods, or recentering never jumps the cursor. Eye gaze is not offered as an input in this version.
- Head tracking supports one-shot left-click dwell. When Windows exposes a control, dwell follows that control and restarts if the pointer moves to a different control; otherwise it uses a small screen-position radius. AccessFlow attempts a Windows UI Automation activation at completion, with a coordinate click fallback. Hand dwell uses the same target change reset. The live session shows a progress ring; the Tracking preview never clicks. Right-click remains available through Voice.
- Local voice commands and dictation use Vosk. Use the Download small English model button in Input methods, then enable voice. The model is stored under your local app data.
- Supported voice commands include left click, right click, scroll up/down, open Notepad/Calculator, and “type …” dictation. “Pause AccessFlow” stops all active providers. The Accessibility page has a built-in keyboard for hand and head pointer users.
- All events pass through the confidence/source gate before reaching WindowsActions.
- pywinauto is used for named control clicks and window activation; PyAutoGUI provides coordinate mouse/keyboard actions. The AccessFlow on-screen keyboard stays open without taking text focus, so typing does not depend on Windows’ On-Screen Keyboard.
- Escape is observed system-wide, so it pauses a running session even when another app has focus. The key is not consumed, so the focused app still receives it normally. Input aimed at elevated (administrator) windows cannot be observed. If Windows refuses to install the global hook (group policy, exhausted hook quota), AccessFlow says so in the session status, on the dashboard, and in a tray notification instead of quietly running without it; Escape then still pauses while an AccessFlow window has focus.
- The touchpad and mouse are never disabled. When real touchpad or mouse movement is detected, camera pointer output steps aside for 1.5 seconds, then resumes automatically.
- The Tracking preview page shows the live mirrored camera, its landmarks, and a miniature screen with the current mapping and smoothing applied. It never moves the real pointer, so hand and head mapping can be checked safely before starting a session.
- Settings persist locally with QSettings. AccessFlow does not record raw audio or video.

Head tracking learns a neutral position at session start and does not require eye calibration. Eye gaze and its calibration flow are not exposed in this version. Providers are opt-in. Use Escape from any application or Emergency pause to stop them; the UI pauses immediately and camera/audio workers finish shutting down in the background.


## Input stability

- Hand control is a relative trackpad mapping, not absolute fingertip aiming: only movement while the hand is visible moves the pointer. A fingertip EMA (0.65) reduces landmark jitter; 1.8× sensitivity, a 0.012 dead zone without reference leak, and a per-frame 180 px cap balance response and cursor stability. The cursor pauses while a dwell click is armed. Keep only the index fingertip still for a left click; the other fingers may be relaxed. Raise only index and middle fingers and keep the index fingertip still for a right click. Pinching is excluded from click recognition; move away after a click to re-arm. For head and hand, dwell restarts when a Windows UI Automation control changes. If an app does not expose controls, dwell uses a small screen-position radius instead. The pointer freezes while the hand leaves the frame and only re-anchors on the next detection, so a hand returning elsewhere never teleports the cursor. Head tracking uses a plain EMA on absolute position; brief hand-tracking dropouts do not immediately switch control to face tracking.
- Voice commands use Vosk word confidence and discard low-confidence recognition. Dictation is scored on the average word instead, because one shaky word should not discard a whole sentence.
- Typing and UI Automation target activation run on a dedicated action thread, so dictation and control lookup cannot freeze the window. Pointer movement, direct clicks, and scroll stay synchronous because they take milliseconds.
- Pausing cancels that work: the action backlog is dropped and a phrase that is already being typed stops within about 64 ms (32 characters) instead of typing to the end. Quitting waits for the action thread, so the process never exits halfway through a keystroke and leaves a modifier held.
- Voice audio is buffered to a bounded queue; if recognition falls behind realtime, audio is dropped and reported instead of growing without limit.

Syntax, offscreen UI construction, and automated regression tests have been validated. A live Windows run is still needed to confirm camera tracking, Windows UI Automation across apps, keyboard focus preservation, and microphone/voice behaviour on this device.

## Tests

    .venv\Scripts\python.exe -m unittest discover -s tests -v

`tests/test_safety.py` covers the pause path (backlog dropped, typing stopped mid-phrase, thread joined on quit), the confidence gate per action, disabled sources, failed-action reporting, and a failed hook install being reported. `tests/test_calibration.py` and the gaze-mapping cases in `tests/test_mapping.py` cover retained legacy calibration math; gaze is not exposed in the current input flow. Other mapping tests cover still-hand stability, reacquisition, and stable head-neutral learning. These tests do not need a camera, microphone, or person.

- Camera output coordinates use PyAutoGUI’s physical-pixel screen size, matching Windows display scaling.

## HCI evaluation

The current prototype's task-based usability evaluation plan and a coded-results template are in [`docs/HCI_EVALUATION.md`](docs/HCI_EVALUATION.md) and [`docs/evaluation_results_template.csv`](docs/evaluation_results_template.csv). The plan focuses on observed usability of the existing workflows; it does not claim proven accessibility benefit.
