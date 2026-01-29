import cv2
import numpy as np

def detect_lanes(frame):
    height, width = frame.shape[:2]
    
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    
    lower_white = np.array([0, 190, 0]) 
    upper_white = np.array([180, 255, 255])
    mask_white = cv2.inRange(hls, lower_white, upper_white)
    
    mask = np.zeros_like(mask_white)
    
    roi_points = np.array([[
        (int(width * 0.15), height),
        (int(width * 0.46), int(height * 0.65)),
        (int(width * 0.54), int(height * 0.65)),
        (int(width * 0.85), height)
    ]], np.int32)
    
    cv2.fillPoly(mask, roi_points, 255)
    roi_mask = cv2.bitwise_and(mask_white, mask)
    
    kernel = np.ones((5, 5), np.uint8)
    roi_mask = cv2.dilate(roi_mask, kernel, iterations=1)
    
    # Пошук ліній
    lines = cv2.HoughLinesP(
        roi_mask, 
        rho=2, 
        theta=np.pi/180, 
        threshold=40, 
        minLineLength=60, 
        maxLineGap=150
    )
    
    return lines