from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Change index if needed (0, 1, 2)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Yield the frame as an HTTP response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
