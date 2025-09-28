import cv2
import mediapipe as mp
import numpy as np

def simple_gesture_demo():
    print("Simple Gesture Demo")
    print("Point your index finger up to trigger 'jump'")
    print("Press 'q' to quit")
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(rgb_frame)
        
        gesture = "idle"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                landmarks = hand_landmarks.landmark
                
                index_tip_y = landmarks[8].y
                index_pip_y = landmarks[6].y
                
                if index_tip_y < index_pip_y:
                    gesture = "jump"
        
        cv2.putText(frame, f"Gesture: {gesture}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        cv2.putText(frame, "Point index finger up = JUMP", (10, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 430), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Simple Gesture Demo', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Demo completed!")

if __name__ == "__main__":
    simple_gesture_demo()