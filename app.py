import cv2
import mediapipe as mp
import numpy as np
import time
import math
import pytesseract
from pathlib import Path

# ---------------- CONFIGURATION ----------------

MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

SMOOTHING = 0.35
PINCH_THRESHOLD = 45
ERASER_RADIUS = 45
CLEAR_HOLD_TIME = 2.0

# ---------------- MEDIAPIPE SETUP ----------------

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

# ---------------- CAMERA SETUP ----------------

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not camera.isOpened():
    raise RuntimeError("Camera open nahi ho raha.")

canvas = None
previous_point = None
smooth_point = None

palm_start_time = None
clear_triggered = False

frame_number = 0
ocr_text = ""
ocr_message_until = 0

# ---------------- HELPER FUNCTIONS ----------------

def finger_is_up(hand, tip, pip):
    return hand[tip].y < hand[pip].y


def get_finger_states(hand):
    return (
        finger_is_up(hand, 8, 6),
        finger_is_up(hand, 12, 10),
        finger_is_up(hand, 16, 14),
        finger_is_up(hand, 20, 18)
    )


def distance_between(p1, p2, width, height):
    x1, y1 = p1.x * width, p1.y * height
    x2, y2 = p2.x * width, p2.y * height
    return math.hypot(x2 - x1, y2 - y1)


def draw_text(frame, text, position, color=(255, 255, 255)):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA
    )


def recognize_handwriting(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # Convert green ink into a binary image.
    _, binary = cv2.threshold(
        gray, 20, 255, cv2.THRESH_BINARY
    )

    # Black text on a white background.
    binary = cv2.bitwise_not(binary)

    binary = cv2.copyMakeBorder(
        binary, 40, 40, 40, 40,
        cv2.BORDER_CONSTANT,
        value=255
    )

    binary = cv2.resize(
        binary, None,
        fx=2, fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    text = pytesseract.image_to_string(
        binary,
        config="--oem 3 --psm 6"
    )

    return text.strip(), binary


# ---------------- MAIN PROGRAM ----------------

print("ZERO TOUCH STARTED!")
print("PINCH = DRAW | TWO FINGERS = ERASE")
print("OPEN PALM = CLEAR | R = OCR | Q = EXIT")

try:
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:

        while camera.isOpened():

            success, frame = camera.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            if canvas is None:
                canvas = np.zeros_like(frame)

            frame_number += 1
            timestamp_ms = frame_number * 33

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = detector.detect_for_video(
                mp_image, timestamp_ms
            )

            mode = "NO HAND DETECTED"

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                index, middle, ring, pinky = get_finger_states(hand)

                pinch_distance = distance_between(
                    hand[4], hand[8], width, height
                )

                pinch = pinch_distance < PINCH_THRESHOLD

                raw_x = int(hand[8].x * width)
                raw_y = int(hand[8].y * height)

                if smooth_point is None:
                    smooth_point = (raw_x, raw_y)
                else:
                    smooth_x = int(
                        SMOOTHING * raw_x
                        + (1 - SMOOTHING) * smooth_point[0]
                    )
                    smooth_y = int(
                        SMOOTHING * raw_y
                        + (1 - SMOOTHING) * smooth_point[1]
                    )
                    smooth_point = (smooth_x, smooth_y)

                x, y = smooth_point

                # PINCH: DRAW
                if pinch:

                    mode = "PINCH - AIR WRITING"

                    cv2.circle(
                        frame, (x, y), 12,
                        (0, 255, 0), -1
                    )

                    if previous_point is not None:
                        cv2.line(
                            canvas,
                            previous_point,
                            (x, y),
                            (0, 255, 0),
                            5,
                            cv2.LINE_AA
                        )

                    previous_point = (x, y)

                    palm_start_time = None
                    clear_triggered = False

                # OPEN PALM: CLEAR
                elif index and middle and ring and pinky:

                    mode = "OPEN PALM - HOLD TO CLEAR"
                    previous_point = None

                    if palm_start_time is None:
                        palm_start_time = time.monotonic()

                    elapsed = time.monotonic() - palm_start_time
                    progress = min(elapsed / CLEAR_HOLD_TIME, 1.0)

                    cv2.ellipse(
                        frame,
                        (width - 100, 110),
                        (45, 45),
                        -90,
                        0,
                        int(360 * progress),
                        (0, 165, 255),
                        8
                    )

                    if elapsed >= CLEAR_HOLD_TIME and not clear_triggered:
                        canvas[:] = 0
                        clear_triggered = True
                        ocr_text = ""

                # TWO FINGERS: ERASER
                elif index and middle and not ring and not pinky:

                    mode = "TWO FINGERS - ERASER"

                    previous_point = None
                    palm_start_time = None
                    clear_triggered = False

                    cv2.circle(
                        frame,
                        (x, y),
                        ERASER_RADIUS,
                        (255, 0, 255),
                        3
                    )

                    cv2.circle(
                        canvas,
                        (x, y),
                        ERASER_RADIUS,
                        (0, 0, 0),
                        -1
                    )

                # ONE FINGER: CURSOR
                elif index and not middle and not ring and not pinky:

                    mode = "POINT - CURSOR"

                    previous_point = None
                    palm_start_time = None
                    clear_triggered = False

                    cv2.circle(
                        frame, (x, y), 12,
                        (0, 0, 255), -1
                    )

                else:
                    mode = "HAND DETECTED"
                    previous_point = None
                    palm_start_time = None
                    clear_triggered = False

            else:
                previous_point = None
                smooth_point = None
                palm_start_time = None
                clear_triggered = False

            # DISPLAY CANVAS
            output = cv2.addWeighted(
                frame, 1.0, canvas, 0.75, 0
            )

            cv2.rectangle(
                output,
                (0, 0),
                (width, 65),
                (25, 25, 25),
                -1
            )

            draw_text(
                output,
                "ZEROTOUCH | TOUCHLESS HMI",
                (20, 42),
                (0, 255, 255)
            )

            draw_text(
                output,
                "MODE: " + mode,
                (20, 105)
            )

            draw_text(
                output,
                "POINT: Cursor | PINCH: Draw | TWO FINGERS: Erase",
                (20, height - 55)
            )

            draw_text(
                output,
                "OPEN PALM: Clear | R: OCR | Q: Exit",
                (20, height - 20)
            )

            if time.monotonic() < ocr_message_until:
                draw_text(
                    output,
                    "OCR: " + ocr_text[:55],
                    (20, 145),
                    (0, 255, 255)
                )

            cv2.imshow(
                "ZeroTouch - Gesture Controlled Interface",
                output
            )

            # KEYBOARD CONTROLS
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):

                print("\n--- ZEROTOUCH OCR RESULT ---")

                try:
                    recognized, processed = recognize_handwriting(canvas)

                    ocr_text = recognized if recognized else "No text detected"

                    print(ocr_text)

                    cv2.imshow("OCR Processed Image", processed)

                except Exception as error:
                    ocr_text = "OCR ERROR"
                    print("OCR Error:", error)

                ocr_message_until = time.monotonic() + 5

            if key == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("ZeroTouch stopped.")
