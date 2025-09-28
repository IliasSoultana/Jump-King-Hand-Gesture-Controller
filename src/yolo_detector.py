import cv2
import numpy as np
from typing import List, Tuple, Optional

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available - using MediaPipe only")

class YOLOHandDetector:
    def __init__(self, model_path: str = 'yolov8n.pt'):
        self.yolo_available = YOLO_AVAILABLE
        self.model = None
        
        if self.yolo_available:
            try:
                self.model = YOLO(model_path)
                print("YOLO model loaded successfully")
            except Exception as e:
                print(f"Failed to load YOLO model: {e}")
                self.yolo_available = False
    
    def detect_hands(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if not self.yolo_available or self.model is None:
            return []
        
        try:
            results = self.model(frame, classes=[0])
            hand_boxes = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = box.conf.item()
                        if confidence > 0.5:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            hand_boxes.append((int(x1), int(y1), int(x2), int(y2)))
            
            return hand_boxes
        
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []
    
    def draw_detections(self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        result_frame = frame.copy()
        
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_frame, 'Hand', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return result_frame
    
    def is_available(self) -> bool:
        return self.yolo_available and self.model is not None