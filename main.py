import cv2 as cv
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras_facenet import FaceNet


# Initialization
facnet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")

Y = faces_embeddings["arr_1"]
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

model = pickle.load(open("svm_model_160x160.pkl", "rb"))

cap = cv.VideoCapture()


while cap.isOpened():
    _, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    for x, y, w, h in faces:
        img = rgb_img[y : y + h, x : x + w]
        img = cv.resize(img, (160, 160))  # 160x160x3
        # img =
