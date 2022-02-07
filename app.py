from flask import Flask, render_template, Response
import cv2
import numpy as np
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
app = Flask(__name__)


def emotion():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5)
            for x, y, w, h in face:
                img = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
                try:
                    analyze = DeepFace.analyze(frame, actions=['emotion'])
                    # print(analyze['dominant_emotion'])
                    cv2.putText(frame, analyze['dominant_emotion'], (x-30,
                                y-75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                except:
                    # print('No Face')
                    cv2.putText(frame, 'No Face', (x-30,
                                y-75), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)

                ret, buffer = cv2.imencode('.jpg', frame)

                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
