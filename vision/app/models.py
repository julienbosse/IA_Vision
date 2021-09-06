import cv2 as cv


def gen_frames_detection():  
    camera = cv.VideoCapture(0)
    face_cascade = cv.CascadeClassifier('app/static/haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('app/static/haarcascade_eye.xml')
    while True:
        success, frame = camera.read()  # read the camera frame
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 8)
        if not success:
            break
        else:
            try:
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