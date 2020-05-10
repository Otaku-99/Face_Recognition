import cv2
import numpy as np
from os import path

name = input("Enter your name : ")

cap = cv2.VideoCapture(0)


classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

count = 5

face_list = []

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = classifier.detectMultiScale(gray)

    areas = []

    for face in faces:
        x, y, w, h = face
        area = w*h
        areas.append((area, face))

    # append area and dace coordinates
    areas = sorted(areas, reverse=True)

    if len(areas) > 0:
        face = areas[0][1]
        x, y, w, h = face
        # Gray image used to recognize image 2D
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))
        cv2.imshow("video", face_img)
        face_list.append(face_img.flatten())
        count-=1
        print("Loaded", 5-count)
        if count <= 0:
            break
    if cv2.waitKey(1) > 30:
        break

face_list = np.array(face_list)
name_list = np.full((len(face_list), 1), name)

total = np.hstack([name_list, face_list])

if path.exists("Image.npy"):
    data = np.load("image.npy")
    data = np.vstack([data, total])
else:
    data = total

np.save("Image.npy", data)

cap.release()
cv2.destroyAllWindows()