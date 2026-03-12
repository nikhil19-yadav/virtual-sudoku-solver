# camera_tracking.py


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
    cap = cv2.VideoCapture(0)  # open webcam

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("⚠️ Failed to access camera.")
            break

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)

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

                # Draw a circle at fingertip
                cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)

                cv2.putText(frame, f"Finger: ({index_x}, {index_y})", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show the frame
        cv2.imshow("Hand Tracking", frame)

        # Press ESC to exit
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
