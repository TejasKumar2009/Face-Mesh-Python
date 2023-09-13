import cv2 as cv
import mediapipe as mp

capture = cv.VideoCapture(0)

mp_facemesh = mp.solutions.face_mesh
facemesh = mp_facemesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

while True:
    isTrue, frame = capture.read()

    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = facemesh.process(frameRGB)

    if (results.multi_face_landmarks):
        for face_landmark in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmark, mp_facemesh.FACEMESH_CONTOURS, mp_drawing.DrawingSpec(color=(0,255,0), thickness=2))

    cv.imshow("Face Mesh", frame)
    
    if cv.waitKey(1) == 27:
        break


capture.release()
cv.destroyAllWindows()
