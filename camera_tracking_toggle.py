# camera_tracking_toggle.py


import cv2
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

def main():
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("⚠️ Failed to open camera.")
        return

    tracking_enabled = False  # Start with only camera feed

    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️ Failed to capture frame.")
            break

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)

        if tracking_enabled:
            # Convert to RGB for Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Draw landmarks if hand detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    # Extract index finger tip coordinates
                    h, w, _ = frame.shape
                    index_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
                    index_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)

                    # Draw fingertip circle
                    cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)

                    # Display coordinates
                    cv2.putText(frame, f"Finger: ({index_x}, {index_y})",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 2)

        # Show the frame
        mode_text = "Tracking ON" if tracking_enabled else "Tracking OFF"
        cv2.putText(frame, mode_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255) if tracking_enabled else (200, 200, 200), 2)

        cv2.imshow("Camera / Hand Tracking", frame)

        # Key controls
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC to quit
            break
        elif key == ord('t'):  # Toggle tracking
            tracking_enabled = not tracking_enabled

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
