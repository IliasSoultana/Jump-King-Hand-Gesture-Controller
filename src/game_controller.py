import time
from typing import Dict, Any
import pynput
from pynput import keyboard

class GameController:
    def __init__(self):
        self.keyboard_controller = pynput.keyboard.Controller()
        
        self.key_mapping = {
            'jump': keyboard.Key.space,
            'left': 'a',
            'right': 'd',
            'idle': None
        }
        
        self.last_key_press = {}
        self.debounce_time = 0.2
        self.key_press_duration = 0.1
        
        self.current_gesture = 'idle'
        
        print("Game Controller initialized!")
        print("Key mapping:", {k: str(v) for k, v in self.key_mapping.items()})
    
    def process_gesture(self, gesture: str, confidence: float = 1.0):
        self.current_gesture = gesture
        
        if gesture in self.key_mapping and self.key_mapping[gesture] is not None:
            key = self.key_mapping[gesture]
            
            if self._should_execute_key_press(str(key)):
                self._execute_key_press(key)
                print(f"Executed: {gesture} -> {key} (confidence: {confidence:.2f})")
    
    def _should_execute_key_press(self, key: str) -> bool:
        current_time = time.time()
        
        if key not in self.last_key_press:
            return True
        
        time_since_last = current_time - self.last_key_press[key]
        return time_since_last >= self.debounce_time
    
    def _execute_key_press(self, key):
        try:
            current_time = time.time()
            
            self.keyboard_controller.press(key)
            time.sleep(self.key_press_duration)
            self.keyboard_controller.release(key)
            
            self.last_key_press[str(key)] = current_time
            
        except Exception as e:
            print(f"Error executing key press '{key}': {e}")
    
    def get_status(self) -> Dict:
        return {
            'current_gesture': self.current_gesture,
            'key_mapping': {k: str(v) for k, v in self.key_mapping.items()},
            'debounce_time': self.debounce_time
        }