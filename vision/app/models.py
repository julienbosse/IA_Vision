import cv2 as cv
import time


def gen_frames_detection():
    camera = cv.VideoCapture(0)
    face_cascade = cv.CascadeClassifier('app/static/haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('app/static/haarcascade_eye.xml')
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            try:
                gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 8)
                for (x, y, w, h) in faces:
                    # dessiner le rectangle sur l'image principale
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    eyes  = eye_cascade.detectMultiScale(gray, 1.1, 8)
                    for (ex, ey, ew, eh) in eyes:
                        cv.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)
            except:
                pass

            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def gen_frames_inversion():
    camera = cv.VideoCapture(0)
    face_cascade = cv.CascadeClassifier('app/static/haarcascade_frontalface_default.xml')
    while True:
        success, frame = camera.read()
        
        if not success:
            break
        else:
            try:
                gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 8)
                x1,y1,w1,h1 = faces[0]
                x2,y2,w2,h2 = faces[1]

                face1 = frame[y1:y1+h1, x1:x1+w1]
                face2 = frame[y2:y2+h2, x2:x2+w2]

                face1 = cv.resize(face1, (w2,h2), interpolation= cv.INTER_LINEAR)
                face2 = cv.resize(face2, (w1,h1), interpolation= cv.INTER_LINEAR)

                frame[y1:y1+h1, x1:x1+w1] = face2
                frame[y2:y2+h2, x2:x2+w2] = face1
                
            except:
                pass

            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result