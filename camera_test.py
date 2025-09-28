import cv2

def test_cameras():
    print("Testing camera indices...")
    working_cameras = []
    
    for i in range(5):
        print(f"Testing camera index {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Camera {i} is working!")
                working_cameras.append(i)
                
                cv2.imshow(f'Camera {i} Test', frame)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
            else:
                print(f"Camera {i} opened but can't read frames")
        else:
            print(f"Camera {i} failed to open")
        
        cap.release()
    
    if working_cameras:
        print(f"\nWorking cameras found: {working_cameras}")
        print(f"Use camera index {working_cameras[0]} for the demo")
        return working_cameras[0]
    else:
        print("\nNo working cameras found!")
        print("Make sure your camera is connected and not being used by another app")
        return None

if __name__ == "__main__":
    best_camera = test_cameras()
    
    if best_camera is not None:
        print(f"\nTesting camera {best_camera} for 5 seconds...")
        cap = cv2.VideoCapture(best_camera)
        
        for i in range(150):
            ret, frame = cap.read()
            if ret:
                cv2.putText(frame, f"Camera {best_camera} - Press ESC to exit", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Camera Test', frame)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Camera test completed!")