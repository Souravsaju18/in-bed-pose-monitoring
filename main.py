import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import matplotlib.pyplot as plt
import pyttsx3

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ----------------------------------------
# Mediapipe Setup
# ----------------------------------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False)

# ----------------------------------------
# Voice Engine
# ----------------------------------------
engine = pyttsx3.init()
last_alert_spoken = ""

def speak(text):
    global last_alert_spoken
    if text != last_alert_spoken:
        engine.say(text)
        engine.runAndWait()
        last_alert_spoken = text

# ----------------------------------------
# Progress Bar
# ----------------------------------------
def draw_bar(panel, x, y, value, label, color):
    bar_width = 200
    bar_height = 15
    filled = int(bar_width * value)

    cv2.rectangle(panel, (x, y), (x + bar_width, y + bar_height), (80, 80, 80), -1)
    cv2.rectangle(panel, (x, y), (x + filled, y + bar_height), color, -1)

    cv2.putText(panel, f"{label}: {int(value * 100)}%",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# ----------------------------------------
# Variables
# ----------------------------------------
frame_delay = 0.04
last_posture = None
posture_frames = 0

prediction_buffer = deque(maxlen=5)

movement_count = 0
last_posture_for_movement = None

movement_log = []
time_log = []
frame_count = 0

# 🔥 FALL DETECTION VARIABLES
prev_center_y = None
fall_detected = False

pose_time = {
    "Supine": 0,
    "Left Side": 0,
    "Right Side": 0,
    "Unknown": 0
}
pose_sequence = []

# ----------------------------------------
# Extract Landmarks
# ----------------------------------------
def extract_landmarks(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if not results.pose_landmarks:
        return None, image

    landmarks = []
    for lm in results.pose_landmarks.landmark[:33]:
        landmarks.append(lm.x)
        landmarks.append(lm.y)

    mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return np.array(landmarks), image

# ----------------------------------------
# Classifier
# ----------------------------------------
def classify_posture(features):
    try:
        lsx, lsy = features[22], features[23]
        rsx, rsy = features[24], features[25]

        shoulder_diff = abs(lsy - rsy)
        body_width = abs(lsx - rsx)

        if shoulder_diff > 0.04:
            return ("Right Side", 94) if lsx > rsx else ("Left Side", 94)

        if body_width < 0.15:
            return ("Right Side", 90) if lsx > rsx else ("Left Side", 90)

        return "Supine", 95

    except:
        return "Unknown", 0

# ----------------------------------------
# Video Input
# ----------------------------------------
cap = cv2.VideoCapture("bedtest.mp4")

if not cap.isOpened():
    print("Error opening video")
    exit()

# ----------------------------------------
# MAIN LOOP
# ----------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    features, frame = extract_landmarks(frame)

    posture = "Unknown"
    confidence = 0

    # 🔥 FALL DETECTION LOGIC
    if features is not None:
        # body center (hip approx)
        center_y = (features[47] + features[49]) / 2

        if prev_center_y is not None:
            speed = abs(center_y - prev_center_y)

            if speed > 0.08:  # threshold
                fall_detected = True
                speak("Fall detected")

        prev_center_y = center_y

    if features is not None:
        raw_posture, confidence = classify_posture(features)
        prediction_buffer.append(raw_posture)
        posture = max(set(prediction_buffer), key=prediction_buffer.count)

    # Movement tracking
    if posture != last_posture_for_movement:
        movement_count += 1
        last_posture_for_movement = posture

    # Duration tracking
    if posture != last_posture:
        last_posture = posture
        posture_frames = 0

    posture_frames += 1
    duration = posture_frames * frame_delay

    mins = int(duration) // 60
    secs = int(duration) % 60

    # Report tracking
    pose_time[posture] += frame_delay

    if len(pose_sequence) == 0 or pose_sequence[-1] != posture:
        pose_sequence.append(posture)

    # Scores
    puri = min(duration / 10, 1.0)
    rci = min(movement_count / 5, 1.0)
    cqs = (1 - puri + rci) / 2

    # Alerts
    alert_text = ""

    if fall_detected:
        alert_text = "🚨 FALL DETECTED!"

    elif puri > 0.8:
        alert_text = "🚨 HIGH RISK!"
        speak("High risk detected")

    elif puri > 0.5:
        alert_text = "⚠️ MEDIUM RISK"
        speak("Medium risk")

    # Graph logging
    frame_count += 1
    if frame_count % 5 == 0:
        movement_log.append(movement_count)
        time_log.append(frame_count * frame_delay)

    # ----------------------------------------
    # Title Header
    # ----------------------------------------
    cv2.putText(frame,
                "REAL-TIME HEALTH MONITOR",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2)

    # UI PANEL
    h, w, _ = frame.shape
    panel = np.zeros((200, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    status_color = (0, 255, 0)

    if fall_detected:
        status_color = (0, 0, 255)

    elif puri > 0.5:
        status_color = (0, 165, 255)

    if puri > 0.8:
        status_color = (0, 0, 255)

    cv2.rectangle(panel, (0, 0), (w, 10), status_color, -1)

    cv2.putText(panel, f"Posture: {posture}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    cv2.putText(panel, f"Duration: {mins:02d}:{secs:02d}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if alert_text:
        cv2.putText(panel, alert_text, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    draw_bar(panel, 350, 40, puri, "Risk", (0, 0, 255))
    draw_bar(panel, 350, 80, rci, "Comfort", (0, 255, 255))
    draw_bar(panel, 350, 120, cqs, "Care", (0, 255, 0))

    final_frame = np.vstack([frame, panel])
    cv2.imshow("Elderly Pose Monitoring - Video", final_frame)

    if cv2.waitKey(30) == 27:
        break


# ----------------------------------------
# SAVE GRAPH
# ----------------------------------------
plt.figure()
plt.plot(time_log, movement_log)
plt.xlabel("Time (seconds)")
plt.ylabel("Movement Count")
plt.title("Patient Movement Over Time")

graph_path = "movement_graph.png"
plt.savefig(graph_path)
plt.close()

# ----------------------------------------
# REPORT SCREEN
# ----------------------------------------
total_time = sum(pose_time.values())

report_screen = np.zeros((500, 800, 3), dtype=np.uint8)
report_screen[:] = (30, 30, 30)

y = 60

cv2.putText(report_screen, "PATIENT REPORT", (250, y),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
y += 60

cv2.putText(report_screen, f"Total Time: {total_time:.2f} sec", (50, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
y += 40

for p, t in pose_time.items():
    cv2.putText(report_screen, f"{p}: {t:.2f} sec", (70, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 30

y += 20
cv2.putText(report_screen, f"Movements: {movement_count}", (50, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

cv2.putText(report_screen, "Press D to Download PDF | ESC to Exit",
            (120, 450),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("Report Summary", report_screen)

# ----------------------------------------
# WAIT FOR INPUT
# ----------------------------------------
while True:
    key = cv2.waitKey(0)

    if key == ord('d') or key == ord('D'):
        print("Generating PDF...")

        doc = SimpleDocTemplate("patient_report.pdf")
        styles = getSampleStyleSheet()

        content = []

        content.append(Paragraph(
            "<b><font size=18 color='blue'>Patient Monitoring Report</font></b>",
            styles["Title"]
        ))
        content.append(Spacer(1, 15))

        content.append(Paragraph("<b>Summary</b>", styles["Heading2"]))
        content.append(Spacer(1, 10))

        content.append(Paragraph(
            f"Total Monitoring Time: <b>{total_time:.2f} seconds</b>",
            styles["Normal"]
        ))

        content.append(Paragraph(
            f"Total Movements: <font color='red'><b>{movement_count}</b></font>",
            styles["Normal"]
        ))

        content.append(Spacer(1, 15))

        content.append(Paragraph("<b>Pose Duration Analysis</b>", styles["Heading2"]))
        content.append(Spacer(1, 10))

        table_data = [["Posture", "Duration (sec)"]]

        for p, t in pose_time.items():
            table_data.append([p, f"{t:.2f}"])

        table = Table(table_data, colWidths=[200, 150])

        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        content.append(table)
        content.append(Spacer(1, 20))

        content.append(Paragraph("<b>Posture Sequence</b>", styles["Heading2"]))
        content.append(Spacer(1, 10))

        content.append(Paragraph(
            " → ".join(pose_sequence),
            styles["Normal"]
        ))

        content.append(Spacer(1, 20))

        content.append(Paragraph("<b>Movement Trend</b>", styles["Heading2"]))
        content.append(Spacer(1, 10))

        content.append(Image(graph_path, width=450, height=220))

        doc.build(content)

        print("✅ PDF Downloaded: patient_report.pdf")
        break

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
