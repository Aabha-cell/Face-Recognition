import cv2
import face_recognition
from datetime import datetime
import numpy as np
import os

#function for marking attendance of recognized students
def markAttendance(name):
    with open('AttendanceSheet.csv','r+') as f:
        AttendanceList = f.readlines()
        nameList = []
        for line in AttendanceList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList: #checking if the student isn't already present
            now = datetime.now()
            time = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{time}')

# reading known student names and images from students folder and storing into respective lists
path = 'students'
images = []
names = []
class_list = os.listdir(path)
for student in class_list:
    curImg = cv2.imread(f'{path}/{student}')
    images.append(curImg)
    names.append(os.path.splitext(student)[0])

# function for finding face encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# calling the findEncodings function to calculate face encodings for known images
encodeListKnown = findEncodings(images)
print("encoding complete")

# accessing the webcam
video = cv2.VideoCapture(0)

while True:
    success, frame = video.read()
    frameSmall = cv2.resize(frame,(0,0),None,0.25,0.25)  #downsizing the image
    frameSmall = cv2.cvtColor(frameSmall, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(frameSmall) #list of location of all faces found in current frame
    encodesCurFrame = face_recognition.face_encodings(frameSmall,facesCurFrame) #list of encodings of all faces found in current frame

    for encodeFace, faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)#comparing encoding of current face with encodings of known faces
        distance = face_recognition.face_distance(encodeListKnown,encodeFace)#distance between encoding of current face with encodings of known faces
        matchIndex = np.argmin(distance)# minimum distance index

        if matches[matchIndex]:
            name = names[matchIndex]
            y1,x2,y2,x1 = faceLoc #coordinates of matched face
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4 #upsizing
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2) # draw rectangle around face
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),2)
            markAttendance(name) #calling the function for marking attendance

    cv2.imshow('Webcam', frame)
    cv2.waitKey(1)
    

