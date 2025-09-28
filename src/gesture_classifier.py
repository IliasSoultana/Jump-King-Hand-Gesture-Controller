import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Optional, Tuple
import os

class GestureClassifier:
    def __init__(self, model_path: str = 'models/gesture_model.pkl'):
        self.model_path = model_path
        self.scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.gesture_classes = ['idle', 'jump', 'left', 'right']
        self.gesture_history: List[str] = []
        self.smoothing_window = 5
        self.load_model()
    
    def create_sample_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(42)
        n_samples = 400
        n_features = 13
        
        X = []
        y = []
        
        for gesture_idx, gesture in enumerate(self.gesture_classes):
            for _ in range(n_samples // len(self.gesture_classes)):
                if gesture == 'idle':
                    features = np.random.normal(0.3, 0.1, n_features)
                elif gesture == 'jump':
                    features = np.random.normal(0.1, 0.05, n_features)
                elif gesture == 'left':
                    features = np.random.normal(0.2, 0.1, n_features)
                    features[5] = -1.5
                else:
                    features = np.random.normal(0.2, 0.1, n_features)
                    features[5] = 1.5
                
                X.append(features)
                y.append(gesture)
        
        return np.array(X), np.array(y)
    
    def train_model(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        if X is None or y is None:
            print("No training data provided, creating sample data...")
            X, y = self.create_sample_training_data()
        
        print(f"Training model with {len(X)} samples...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained! Accuracy: {accuracy:.3f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        self.save_model()
    
    def predict_gesture(self, features: np.ndarray, return_confidence: bool = False) -> str:
        if self.model is None or self.scaler is None:
            if not self.load_model():
                self.train_model()
        
        if len(features) == 0:
            return 'idle'
        
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            prediction = self.model.predict(features_scaled)[0]
            confidence = np.max(self.model.predict_proba(features_scaled))
            
            if confidence < 0.6:
                prediction = 'idle'
            
            smoothed_gesture = self.smooth_prediction(prediction)
            
            if return_confidence:
                return smoothed_gesture, confidence
            return smoothed_gesture
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 'idle'
    
    def smooth_prediction(self, prediction: str) -> str:
        self.gesture_history.append(prediction)
        
        if len(self.gesture_history) > self.smoothing_window:
            self.gesture_history.pop(0)
        
        if len(self.gesture_history) >= 3:
            gesture_counts = {}
            for gesture in self.gesture_history:
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
            return max(gesture_counts, key=gesture_counts.get)
        
        return prediction
    
    def save_model(self):
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            if self.model is not None:
                joblib.dump(self.model, self.model_path)
                print(f"Model saved to {self.model_path}")
            
            if self.scaler is not None:
                joblib.dump(self.scaler, self.scaler_path)
                print(f"Scaler saved to {self.scaler_path}")
                
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self) -> bool:
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                print("Pre-trained model loaded successfully!")
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
        
        return False