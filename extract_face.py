import cv2
import streamlit as st
# Read the input image
def extract_face(img):  
    # Convert into grayscale
    output = []
    rectangle = []
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(img, 1.05, minNeighbors = 4)
    for (x, y, w, h) in faces:
        rectangle.append([x,y,w,h])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
        faces = img[y:y + h, x:x + w]
        faces = cv2.resize(faces, (35,35))
        output.append(faces)
    return output,rectangle