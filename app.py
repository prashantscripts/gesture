from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp


app = Flask(__name__)

@app.route('/test')
def test():
    return "Flask is working!"

# Load the trained model
model = tf.keras.models.load_model("asl_model.h5")

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

# Define gesture labels (must match training labels)
gesture_labels = ["A", "B", "C", "D", "E"]  # Update based on your dataset

def predict_gesture(landmarks):
    """ Predicts gesture based on hand landmarks. """
    landmarks = np.array(landmarks).reshape(1, -1)  # Reshape for model
    prediction = model.predict(landmarks)
    return gesture_labels[np.argmax(prediction)]  # Get the highest probability gesture

def generate_frames():
    """ Captures video frames and processes hand gestures in real-time. """
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        detected_gesture = "None"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmark positions
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                detected_gesture = predict_gesture(landmarks)

        # Display detected gesture
        cv2.putText(frame, f"Gesture: {detected_gesture}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)



