import cv2
import mediapipe as mp
import numpy as np
import csv

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

gesture_name = input("Enter gesture name: ")  # Type 'A', 'B', etc.

with open("gesture_data.csv", "a", newline="") as f:
    writer = csv.writer(f)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                landmarks.append(gesture_name)  # Add label to the data
                writer.writerow(landmarks)

        cv2.imshow("Collecting Data", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop
            break

cap.release()
cv2.destroyAllWindows()
