
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def detect_hands(self, frame: np.ndarray) -> Tuple[np.ndarray, List[List[float]]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        landmarks_list = []
        annotated_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    annotated_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                landmarks_list.append(landmarks)
        
        return annotated_frame, landmarks_list
    
    def extract_features(self, landmarks: List[float]) -> np.ndarray:
        if len(landmarks) < 63:
            return np.array([])
        
        features = []
        
        points = [(landmarks[i], landmarks[i+1], landmarks[i+2]) 
                 for i in range(0, len(landmarks), 3)]
        
        wrist = points[0]
        thumb_tip = points[4]
        index_tip = points[8]
        middle_tip = points[12]
        ring_tip = points[16]
        pinky_tip = points[20]
        
        for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]:
            distance = np.sqrt(
                (tip[0] - wrist[0])**2 + 
                (tip[1] - wrist[1])**2 + 
                (tip[2] - wrist[2])**2
            )
            features.append(distance)
        
        for tip in [index_tip, middle_tip, ring_tip, pinky_tip]:
            angle_x = np.arctan2(tip[1] - wrist[1], tip[0] - wrist[0])
            angle_y = np.arctan2(tip[2] - wrist[2], tip[0] - wrist[0])
            features.extend([angle_x, angle_y])
        
        return np.array(features)
    
    def cleanup(self):
        self.hands.close()