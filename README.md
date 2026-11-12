# ZeroTouch

A webcam-based touchless gesture-controlled drawing interface built with Python, OpenCV, MediaPipe Tasks, NumPy, and Tesseract OCR.

## Features

- Index finger: move the on-screen cursor.
- Thumb-index pinch: draw on the virtual canvas.
- Two fingers: erase.
- Open palm held for 2 seconds: clear the canvas.
- Press `R`: run Tesseract OCR on the current canvas.
- Press `Q`: quit.

## Requirements

- Python 3.11 recommended.
- A webcam.
- Tesseract OCR installed on Windows.
- MediaPipe Hand Landmarker model file.

## Setup (Windows PowerShell)

1. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install Python dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Download the MediaPipe Hand Landmarker model from the official MediaPipe documentation/model-bundle source and save it as:

   ```
   models/hand_landmarker.task
   ```

   Create the `models` folder if it does not exist.

4. Install Tesseract OCR for Windows. The current `app.py` expects it at:

   ```
   C:\Program Files\Tesseract-OCR\tesseract.exe
   ```

   If installed elsewhere, update `pytesseract.pytesseract.tesseract_cmd` in `app.py`.

5. Run:

   ```powershell
   python app.py
   ```

## Controls

| Input | Action |
|---|---|
| Point with index finger | Cursor |
| Pinch thumb and index finger | Draw |
| Raise index and middle fingers | Erase |
| Open palm for 2 seconds | Clear canvas |
| `R` | OCR |
| `Q` | Quit |

This is a laptop webcam prototype. It does not connect to or control a real vehicle.
