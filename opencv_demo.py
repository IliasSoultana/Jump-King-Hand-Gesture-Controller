import cv2
import numpy as np

def detect_simple_gestures(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    gesture = "idle"
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) > 5000:
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            
            hull = cv2.convexHull(largest_contour, returnPoints=False)
            
            if len(hull) > 3:
                defects = cv2.convexityDefects(largest_contour, hull)
                
                if defects is not None:
                    defect_count = 0
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(largest_contour[s][0])
                        end = tuple(largest_contour[e][0])
                        far = tuple(largest_contour[f][0])
                        
                        if d > 10000:
                            defect_count += 1
                            cv2.circle(frame, far, 5, (0, 0, 255), -1)
                    
                    if defect_count == 0:
                        gesture = "jump"
                    elif defect_count >= 4:
                        gesture = "idle"
                    else:
                        moments = cv2.moments(largest_contour)
                        if moments["m00"] != 0:
                            cx = int(moments["m10"] / moments["m00"])
                            frame_center = frame.shape[1] // 2
                            
                            if cx < frame_center - 50:
                                gesture = "left"
                            elif cx > frame_center + 50:
                                gesture = "right"
    
    return frame, gesture, skin_mask

def opencv_gesture_demo():
    print("OpenCV Gesture Demo (No MediaPipe required)")
    print("This works with basic computer vision:")
    print("• Closed fist = JUMP")
    print("• Open palm = IDLE") 
    print("• Move hand left/right for LEFT/RIGHT")
    print("Press 'q' to quit, 's' to show skin mask")
    
    cap = cv2.VideoCapture(0)
    show_mask = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        processed_frame, gesture, skin_mask = detect_simple_gestures(frame)
        
        cv2.putText(processed_frame, f"Gesture: {gesture.upper()}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        cv2.putText(processed_frame, "Fist=Jump, Open=Idle, Move=Left/Right", (10, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(processed_frame, "Press 'q'=quit, 's'=show skin detection", (10, 430), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('OpenCV Gesture Demo', processed_frame)
        
        if show_mask:
            cv2.imshow('Skin Detection', skin_mask)
        else:
            cv2.destroyWindow('Skin Detection')
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_mask = not show_mask
    
    cap.release()
    cv2.destroyAllWindows()
    print("Demo completed!")

if __name__ == "__main__":
    opencv_gesture_demo()