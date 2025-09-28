import cv2
import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from hand_detector import HandDetector
from yolo_detector import YOLOHandDetector  
from gesture_classifier import GestureClassifier
from game_controller import GameController

class JumpKingGestureController:
    def __init__(self):
        print("Initializing Jump King Hand Gesture Controller...")
        
        self.hand_detector = HandDetector()
        self.yolo_detector = YOLOHandDetector()
        self.gesture_classifier = GestureClassifier()
        self.game_controller = GameController()
        
        self.paused = False
        self.current_gesture = 'idle'
        self.confidence = 0.0
        self.gesture_detected = False
    
    def process_frame(self, frame):
        if self.paused:
            return frame
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        hand_results = self.hand_detector.hands.process(rgb_frame)
        
        if self.yolo_detector.is_available():
            yolo_detections = self.yolo_detector.detect_hands(frame)
        
        gesture_detected = False
        current_gesture = 'idle'
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.hand_detector.draw_landmarks(frame, hand_landmarks)
                
                features = self.hand_detector.extract_features(hand_landmarks)
                if features is not None:
                    predicted_gesture = self.gesture_classifier.predict_gesture(features, return_confidence=True)
                    if isinstance(predicted_gesture, tuple):
                        predicted_gesture, confidence = predicted_gesture
                        self.confidence = confidence
                    else:
                        self.confidence = 0.8
                    
                if gesture_detected:
                    current_gesture = predicted_gesture
                    
                    self.game_controller.handle_gesture(current_gesture)
                
                gesture_detected = True
                break
        
        self.current_gesture = current_gesture
        self.gesture_detected = gesture_detected
        return frame
    
    def draw_ui(self, frame, gesture, confidence, fps, detected):
        overlay = frame.copy()
        
        cv2.rectangle(overlay, (10, 10), (400, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_offset = 40
        cv2.putText(frame, f"Gesture: {gesture}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        y_offset += 35
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 30
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 30
        status = "Active" if detected else "Searching..."
        color = (0, 255, 0) if detected else (0, 165, 255)
        cv2.putText(frame, f"Status: {status}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.putText(frame, "Controls:", (20, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "ESC: Quit", (20, 175), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        cv2.putText(frame, "SPACE: Pause/Resume", (20, 195), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        cv2.putText(frame, "R: Retrain model", (20, 215), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
    
    def handle_keyboard_input(self, key):
        if key == 27:
            return False
        elif key == ord(' '):
            self.paused = not self.paused
            print(f"{'Paused' if self.paused else 'Resumed'}")
        elif key == ord('r') or key == ord('R'):
            print("Retraining model...")
            self.gesture_classifier.train_model()
        
        return True

def main():
    print("Initializing gesture controller...")
    controller = JumpKingGestureController()
    
    if not controller.gesture_classifier.model:
        print("Training gesture model...")
        controller.gesture_classifier.train_model()
    
    print("Starting camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    fps_counter = 0
    start_time = time.time()
    current_fps = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = controller.process_frame(frame)
            
            controller.draw_ui(processed_frame, controller.current_gesture, 
                              controller.confidence, current_fps, 
                              controller.gesture_detected)
            
            fps_counter += 1
            if fps_counter % 30 == 0:
                current_fps = 30.0 / (time.time() - start_time)
                start_time = time.time()
            
            cv2.imshow('Jump King Gesture Controller', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if not controller.handle_keyboard_input(key):
                break
    
    except KeyboardInterrupt:
        print("Stopping...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        controller.hand_detector.cleanup()
        print("Cleanup complete")

if __name__ == "__main__":
    main()