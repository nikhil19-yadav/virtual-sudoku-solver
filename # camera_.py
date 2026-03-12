# camera_.py


import cv2  

def main():
    cap = cv2.VideoCapture(0)  # Open webcam (0 = default camera)

    if not cap.isOpened():
        print("⚠️ Failed to open camera.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️ Failed to capture frame.")
            break

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)

        # Show the live camera feed
        cv2.imshow("Camera Feed", frame)

        # Press ESC to exit
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
