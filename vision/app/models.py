import cv2 as cv
import numpy as np
import dlib
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


def gen_video_inversion_swapping():

    def extract_index_nparray(nparray):
        index = None
        for num in nparray[0]:
            index = num
            break
        return index

    # img = cv2.imread("img/bradley-cooper.jpg")
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # mask1 = np.zeros_like(img_gray)

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1200)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 900)

    seamlessclone = cap

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("app/static/shape_predictor_68_face_landmarks.dat")

    img = cv.imread("app/static/img/bradley-cooper.jpg")
    # img = cv2.imread("img/keanureeves.jpg")
    # img = cv2.imread("img/naruto.jpg")
    # img = cv.imread("app/static/img/Sylvester_Stallone.jpg")
    # img = cv.imread("app/static/img/jackson.png")
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask1 = np.zeros_like(img_gray)

    # Face 1
    faces = detector(img_gray)
    for face in faces:
        landmarks = predictor(img_gray, face)
        landmarks_points1 = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points1.append((x, y))

            # cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        # print('test')

        points = np.array(landmarks_points1, np.int32)
        convexhull1 = cv.convexHull(points)
        # cv2.polylines(img, [convexhull1], True, (255, 0, 0), 3)
        cv.fillConvexPoly(mask1, convexhull1, 255)

        face_image_1 = cv.bitwise_and(img, img, mask=mask1)

        # Delaunay triangulation
        rect = cv.boundingRect(convexhull1)
        # (x, y, w, h) = rect
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))
        subdiv = cv.Subdiv2D(rect)
        subdiv.insert(landmarks_points1)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        # we get the Landmark points indexes of each triangle
        indexes_triangles = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = extract_index_nparray(index_pt1)
            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = extract_index_nparray(index_pt2)
            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = extract_index_nparray(index_pt3)
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)

            # cv2.line(img, pt1, pt2, (0, 0, 255), 2)
            # cv2.line(img, pt2, pt3, (0, 0, 255), 2)
            # cv2.line(img, pt3, pt1, (0, 0, 255), 2)

    while True:
        success, img2 = cap.read()
        img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        mask2 = np.zeros_like(img2_gray)

        img2_new_face = np.zeros((img2.shape[0], img2.shape[1], img2.shape[2]), np.uint8)
        img2_face_mask = np.zeros_like(img2_gray)

        # print(img2.shape[2])

        faces = detector(img2_gray)

        try:
            # Face 2
            for face in faces:
                landmarks = predictor(img2_gray, face)
                landmarks_points2 = []
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    landmarks_points2.append((x, y))

                    # cv2.circle(img2, (x, y), 3, (0, 0, 255), -1)
            # print('test')

            points = np.array(landmarks_points2, np.int32)
            convexhull2 = cv.convexHull(points)
            # cv2.polylines(img2, [convexhull2], True, (255, 0, 0), 3)
            cv.fillConvexPoly(mask2, convexhull2, 255)

            face_image_2 = cv.bitwise_and(img2, img2, mask=mask2)

            # Delaunay triangulation
            rect = cv.boundingRect(convexhull2)
            # (x, y, w, h) = rect
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))
            subdiv = cv.Subdiv2D(rect)
            subdiv.insert(landmarks_points2)
            triangles = subdiv.getTriangleList()
            triangles = np.array(triangles, dtype=np.int32)

            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])

                # cv2.line(img2, pt1, pt2, (0, 0, 255), 2)
                # cv2.line(img2, pt2, pt3, (0, 0, 255), 2)
                # cv2.line(img2, pt3, pt1, (0, 0, 255), 2)

            # we get the Landmark points indexes of each triangle
            indexes_triangles = []
            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])
                index_pt1 = np.where((points == pt1).all(axis=1))
                index_pt1 = extract_index_nparray(index_pt1)
                index_pt2 = np.where((points == pt2).all(axis=1))
                index_pt2 = extract_index_nparray(index_pt2)
                index_pt3 = np.where((points == pt3).all(axis=1))
                index_pt3 = extract_index_nparray(index_pt3)
                if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                    triangle = [index_pt1, index_pt2, index_pt3]
                    indexes_triangles.append(triangle)

            # Triangulation of both faces
            for triangle_index in indexes_triangles:
                # print("triangle", triangle_index)

                # Triangulation of the first face
                tr1_pt1 = landmarks_points1[triangle_index[0]]
                tr1_pt2 = landmarks_points1[triangle_index[1]]
                tr1_pt3 = landmarks_points1[triangle_index[2]]
                triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

                rect1 = cv.boundingRect(triangle1)
                (x, y, w, h) = rect1
                cropped_triangle = img[y: y + h, x: x + w]
                cropped_tr1_mask = np.zeros((h, w), np.uint8)

                points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                                   [tr1_pt2[0] - x, tr1_pt2[1] - y],
                                   [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

                cv.fillConvexPoly(cropped_tr1_mask, points, 255)
                cropped_triangle = cv.bitwise_and(cropped_triangle, cropped_triangle,
                                                   mask=cropped_tr1_mask)
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # cv2.line(img, tr1_pt1, tr1_pt2, (0, 0, 255), 2)
                # cv2.line(img, tr1_pt2, tr1_pt3, (0, 0, 255), 2)
                # cv2.line(img, tr1_pt3, tr1_pt1, (0, 0, 255), 2)

                # Triangulation of the second face
                tr2_pt1 = landmarks_points2[triangle_index[0]]
                tr2_pt2 = landmarks_points2[triangle_index[1]]
                tr2_pt3 = landmarks_points2[triangle_index[2]]
                triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

                rect2 = cv.boundingRect(triangle2)
                (x, y, w, h) = rect2
                cropped_triangle2 = img2[y: y + h, x: x + w]

                cropped_tr2_mask = np.zeros((h, w), np.uint8)

                points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                    [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                    [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

                cv.fillConvexPoly(cropped_tr2_mask, points2, 255)
                cropped_triangle2 = cv.bitwise_and(cropped_triangle2, cropped_triangle2,
                                                    mask=cropped_tr2_mask)

                # cv2.line(img2, tr2_pt1, tr2_pt2, (0, 0, 255), 2)
                # cv2.line(img2, tr2_pt2, tr2_pt3, (0, 0, 255), 2)
                # cv2.line(img2, tr2_pt3, tr2_pt1, (0, 0, 255), 2)

                # Warp triangles
                points = np.float32(points)
                points2 = np.float32(points2)
                M = cv.getAffineTransform(points, points2)
                warped_triangle = cv.warpAffine(cropped_triangle, M, (w, h))

                # Warp triangles
                warped_triangle = cv.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

                # break

                # Reconstructing destination face
                img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
                img2_new_face_rect_area_gray = cv.cvtColor(img2_new_face_rect_area, cv.COLOR_BGR2GRAY)

                # Let's create a mask to remove the lines between the triangles
                _, mask_triangles_designed = cv.threshold(img2_new_face_rect_area_gray, 1, 255, cv.THRESH_BINARY_INV)
                warped_triangle = cv.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

                img2_new_face_rect_area = cv.add(img2_new_face_rect_area, warped_triangle)
                img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area
                # cv2.imshow("Test", img2_new_face)
                # cv2.waitKey(0)

            # Face swapped (putting 1st face into 2nd face)
            img2_head_mask = cv.fillConvexPoly(img2_face_mask, convexhull2, 255)
            img2_face_mask = cv.bitwise_not(img2_head_mask)
            img2_head_noface = cv.bitwise_and(img2, img2, mask=img2_face_mask)
            result = cv.add(img2_head_noface, img2_new_face)

            (x, y, w, h) = cv.boundingRect(convexhull2)
            center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
            seamlessclone = cv.seamlessClone(result, img2, img2_head_mask, center_face2, cv.MIXED_CLONE)

        except:
            pass
        
        ret, buffer = cv.imencode('.jpg', seamlessclone)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result



