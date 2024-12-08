import cv2 as cv
import numpy as np
import os
from mtcnn import MTCNN  # Import MTCNN
from sklearn.preprocessing import LabelEncoder
from keras_facenet import FaceNet
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")

Y = faces_embeddings["arr_1"]
encoder = LabelEncoder()
encoder.fit(Y)

model = pickle.load(open("svm_model_160x160.pkl", "rb"))

detector = MTCNN()

cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    detections = detector.detect_faces(rgb_img)

    for detection in detections:
        x, y, w, h = detection["box"]
        confidence = detection["confidence"]

        if confidence > 0.90:
            face = rgb_img[y : y + h, x : x + w]
            face = cv.resize(face, (160, 160))
            face = np.expand_dims(face, axis=0)

            ypred = facenet.embeddings(face)
            face_name = model.predict(ypred)
            final_name = encoder.inverse_transform(face_name)[0]

            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.putText(
                frame,
                final_name,
                (x, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )

    cv.imshow("Face Recognition", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
