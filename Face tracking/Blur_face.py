import mediapipe as mp
import cv2
import numpy as np

face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture("Data/video_1.mp4")
_, frame = cap.read()
height, width, _ = frame.shape

# Video writer
writer = cv2.VideoWriter("Data/out.avi", cv2.VideoWriter_fourcc(*'FMP4'),
                         30, (width, height))

with face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        _, frame = cap.read()
        # frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
        frame_copy = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(frame_rgb)
        if result.multi_face_landmarks:
            facelandmarks = []
            for facial_landmarks in result.multi_face_landmarks:
                for i in range(0, 468):
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * width)
                    y = int(pt1.y * height)
                    facelandmarks.append([x, y])
            # Get face landmarks
            facelandmarks = np.array(facelandmarks, np.int32)
            convexhull = cv2.convexHull(facelandmarks)

            # Extract face and blur:
            mask = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(mask, convexhull, 255)
            frame_copy = cv2.blur(frame_copy, (50, 50))
            face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)

            # Extrace background:
            background_mask = cv2.bitwise_not(mask)
            background = cv2.bitwise_and(frame, frame, mask=background_mask)

            # Output
            out = cv2.add(background, face_extracted)
            frame = out

        writer.write(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break

cap.release()
writer.release()
cv2.destroyAllWindows()