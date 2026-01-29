import cv2
import numpy as np
from src.config import *
from src.detector import ObjectDetector
from src.tracker import TrafficTracker

temp_points = []
drawing_mode = None 

def mouse_callback(event, x, y, flags, param):
    global temp_points, drawing_mode, tracker, BLUE_LINE, YELLOW_LINES
    
    if event == cv2.EVENT_RBUTTONDOWN:
        drawing_mode = "blue"
        temp_points = [(x, y)]
    elif event == cv2.EVENT_RBUTTONUP:
        if len(temp_points) == 1:
            temp_points.append((x, y))
            BLUE_LINE = [temp_points[0], temp_points[1]]
            tracker.update_blue_line(BLUE_LINE)
            drawing_mode = None

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_mode = "yellow"
        temp_points = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        if len(temp_points) == 1:
            temp_points.append((x, y))
            YELLOW_LINES.append([temp_points[0], temp_points[1]])
            tracker.add_yellow_line([temp_points[0], temp_points[1]])
            drawing_mode = None

def main():
    global tracker
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    detector = ObjectDetector(MODEL_PATH)
    tracker = TrafficTracker(BLUE_LINE)

    cv2.namedWindow("Traffic System")
    cv2.setMouseCallback("Traffic System", mouse_callback)

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        results = detector.track(frame)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            names = results[0].names

            for box, track_id, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int(y2)
                
                if track_id in tracker.track_history:
                    prev_pos = tracker.track_history[track_id]
                    events = tracker.process_movement(track_id, prev_pos, (cx, cy))
                    
                    if events["crossed"]:
                        print(f"{names[cls]} #{track_id} crossed main line")
                    if events["violation"]:
                        print(f"{names[cls]} #{track_id} LANE VIOLATION")

                tracker.track_history[track_id] = (cx, cy)

                color = (0, 255, 0)
                if any((track_id, i) in tracker.violation_history for i in range(len(tracker.yellow_lines))):
                    color = COLOR_RED
                elif track_id in tracker.crossed_ids:
                    color = COLOR_BLUE

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{names[cls]} #{track_id}", (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.line(frame, tuple(BLUE_LINE[0]), tuple(BLUE_LINE[1]), COLOR_BLUE, 3)
        for y_line in YELLOW_LINES:
            cv2.line(frame, tuple(y_line[0]), tuple(y_line[1]), COLOR_YELLOW, 2)

        cv2.rectangle(frame, (20, 20), (320, 120), (0, 0, 0), -1)
        cv2.putText(frame, f"Crossed: {tracker.total_crossed}", (40, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)
        cv2.putText(frame, f"Violations: {tracker.total_violations}", (40, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 2)

        cv2.imshow("Traffic System", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('c'):
            YELLOW_LINES.clear()
            tracker.clear_yellow_lines()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()