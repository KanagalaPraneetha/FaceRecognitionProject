import cv2
import os
import numpy as np

data_path = 'faces/'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []
    for image_path in image_paths:
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        id = int(os.path.split(image_path)[-1].split(".")[1])  # Extract ID
        faces = detector.detectMultiScale(gray_img)
        for (x, y, w, h) in faces:
            face_samples.append(gray_img[y:y+h, x:x+w])
            ids.append(id)
    return face_samples, np.array(ids)

faces, ids = get_images_and_labels(data_path)
recognizer.train(faces, ids)
recognizer.write('trainer.yml')
print("Model trained and saved!")
