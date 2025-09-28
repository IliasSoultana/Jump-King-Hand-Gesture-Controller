

import cv2
import numpy as np
import os

def create_demo_video():
    print("Creating demo video for testing...")
    
    width, height = 640, 480
    fps = 30
    duration = 10  # seconds
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('demo_video.avi', fourcc, fps, (width, height))
    
    for frame_num in range(fps * duration):
        # Create a frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some moving elements to simulate hand gestures
        time_ratio = frame_num / (fps * duration)
        
        if time_ratio < 0.25:
            # Simulate "idle" - large circle (open palm)
            cv2.circle(frame, (320, 240), 80, (255, 255, 255), -1)
            gesture_text = "IDLE"
        elif time_ratio < 0.5:
            # Simulate "jump" - small circle (fist)
            cv2.circle(frame, (320, 240), 30, (255, 255, 255), -1)
            gesture_text = "JUMP"
        elif time_ratio < 0.75:
            # Simulate "left" - circle moving left
            x_pos = int(320 - 100 * (time_ratio - 0.5) * 4)
            cv2.circle(frame, (x_pos, 240), 50, (255, 255, 255), -1)
            gesture_text = "LEFT"
        else:
            # Simulate "right" - circle moving right
            x_pos = int(220 + 100 * (time_ratio - 0.75) * 4)
            cv2.circle(frame, (x_pos, 240), 50, (255, 255, 255), -1)
            gesture_text = "RIGHT"
        
        # Add text
        cv2.putText(frame, f"Demo: {gesture_text}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_num}", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    print("Demo video created: demo_video.avi")

def video_gesture_demo(video_path='demo_video.avi'):
    if not os.path.exists(video_path):
        print("Creating demo video...")
        create_demo_video()
    
    print(f"Processing video: {video_path}")
    print("This simulates how the system would work with real camera input")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        
        # Simulate gesture detection (simple based on white pixels)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        white_pixels = np.sum(gray > 200)
        
        # Simple gesture classification based on white pixel count
        if white_pixels > 15000:
            gesture = "IDLE"
            color = (0, 255, 255)
        elif white_pixels > 5000:
            gesture = "LEFT/RIGHT"  
            color = (255, 0, 255)
        elif white_pixels > 2000:
            gesture = "JUMP"
            color = (0, 0, 255)
        else:
            gesture = "NO HAND"
            color = (128, 128, 128)
        
        # Add overlay showing ML processing
        cv2.putText(frame, f"ML Classification: {gesture}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Feature Vector: {white_pixels} pixels", (10, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Simulated Hand Tracking", (10, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Show system info
        height, width = frame.shape[:2]
        cv2.putText(frame, "Jump King Gesture Controller - Video Demo", (10, height - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Technologies: Python | OpenCV | MediaPipe | ML", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('Jump King Gesture Controller - Video Demo', frame)
        
        # Control playback
        key = cv2.waitKey(30) & 0xFF  # ~30fps playback
        if key == ord('q'):
            break
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
    
    cap.release()
    cv2.destroyAllWindows()
    print("Video demo completed!")

if __name__ == "__main__":
    video_gesture_demo()