<div align="center">

# AccessFlow

### Alternative computer interaction for an HCI mini-project

**Hand tracking · Head tracking · Dwell selection · Offline voice · On-screen keyboard**

![Platform: Windows](https://img.shields.io/badge/platform-Windows-0078D4?logo=windows&logoColor=white)
![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
![HCI mini-project](https://img.shields.io/badge/project-HCI%20mini--project-6B5B95)

</div>

AccessFlow is a Windows desktop prototype exploring alternative ways to control a computer. It combines webcam-based hand and head pointer control, dwell selection, offline speech input, and an on-screen keyboard in one application.

> **Project status:** Working prototype for demonstration and usability evaluation. Performance can vary with the computer, camera, microphone, lighting, and user setup. AccessFlow is not a certified assistive product.

## Contents

- [What it can do](#what-it-can-do)
- [Voice phrases](#voice-phrases)
- [Requirements](#requirements)
- [Setup](#setup)
- [First use](#first-use)
- [On-screen keyboard](#using-the-on-screen-keyboard)
- [Project structure](#project-structure)
- [Privacy and limitations](#privacy-and-limitations)
- [Tests](#tests)
- [HCI evaluation materials](#hci-evaluation-materials)

## What it can do

- **Hand pointer:** Move the pointer by moving the index finger, as with a trackpad. Hold the index fingertip steady to left-click. Raise and hold the index and middle fingers to right-click. Pinching is not used for clicking.
- **Head pointer:** Move the pointer with head movement. AccessFlow learns a neutral position at the beginning of a session. Hold the pointer steady for a dwell left-click.
- **Adjustable dwell:** Choose a dwell duration from 500 to 3000 milliseconds. A progress indicator shows when a dwell click is about to happen.
- **Offline voice input:** After installing the Vosk speech model, use speech for dictation and supported desktop actions. The model is downloaded once and recognition runs locally.
- **On-screen keyboard:** Open AccessFlow’s keyboard and select its keys with the hand or head pointer, or with a mouse or touchpad.
- **Tracking preview:** Inspect hand or head tracking before starting a session. Preview mode does not move the real pointer or click.
- **Pause controls:** Pause from AccessFlow or press Escape while a session is active.

### Voice phrases

Examples supported by the current voice input include:

| Say | Result |
| --- | --- |
| `type hello there` | Types “hello there” into the focused application |
| `left click` | Sends a left click at the pointer |
| `right click` | Sends a right click at the pointer |
| `scroll up` / `scroll down` | Scrolls at the pointer |
| `open Notepad` / `open Calculator` | Opens and focuses the application |
| `space` / `new line` | Types a space or presses Enter |
| `pause AccessFlow` | Pauses active input methods |

Speech recognition confidence and microphone conditions can affect whether a phrase is accepted. The app does not store raw audio.

## Requirements

- Windows 10 or 11
- Python 3.11 recommended
- A webcam for hand or head tracking
- A microphone for voice input
- Internet access for the first download of the speech model and camera tracking models

## Setup

Open PowerShell in the project folder and run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

If PowerShell blocks virtual-environment activation, run the interpreter directly instead:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

On first use, allow camera and microphone access in Windows privacy settings when prompted.

## First use

1. Open **Tracking preview** and choose **Hand preview** or **Head preview** to check camera tracking.
2. Open **Input methods** and enable the methods you want to use.
3. For voice input, choose **Download small English model** and wait for **Offline voice model ready**. This downloads the Vosk model into your local AccessFlow application data. You can also browse to an already extracted Vosk model folder.
4. Set the dwell duration under **Accessibility** if needed.
5. Start the session. Use **Pause inputs**, **Emergency pause**, or Escape to stop active input.

The hand pointer is relative: move the index finger to steer it across the screen. It does not jump to the screen position your fingertip points at. The head pointer learns its starting neutral position when tracking begins; use **Recenter head tracking** if you change posture.

## Using the on-screen keyboard

Open **Accessibility → Open AccessFlow Keyboard**. Select the text field in the application where you want to type, then choose keys on the AccessFlow keyboard using your active pointer or a mouse/touchpad. The keyboard is built into AccessFlow and does not rely on the Windows On-Screen Keyboard.

## Project structure

```text
AccessFlow/
├── accessflow/
│   ├── core/       # Interaction events and the shared action gate
│   ├── inputs/     # Hand, head, dwell, and voice input workers
│   └── system/     # Windows pointer, keyboard, and application actions
├── tests/          # Automated regression tests
├── docs/           # HCI evaluation materials
├── main.py         # PySide6 desktop interface
└── requirements.txt
```

## Privacy and limitations

- Vosk speech recognition runs locally after the model is downloaded. AccessFlow does not record raw audio or video.
- Webcam tracking requires the camera to remain available to AccessFlow. Another application using the camera may prevent tracking from starting.
- Eye gaze is not available as an input in this version.
- AccessFlow uses Windows desktop input APIs. Elevated applications and secure Windows screens may block normal input, and behavior can vary across applications.
- This prototype has not been established as suitable for every user's access needs. Evaluate it with users and tasks before relying on it.

## Tests

Run the automated regression suite from the project folder:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover input mapping, dwell behavior, voice command routing, action safety, and calibration-related helper code. They do not replace a live camera, microphone, or cross-application evaluation.

## HCI evaluation materials

- [Task-based usability evaluation plan](docs/HCI_EVALUATION.md)
- [Evaluation results template](docs/evaluation_results_template.csv)
