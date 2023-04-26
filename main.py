import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime
import csv


cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://faceattendancerealtime-94252-default-rtdb.firebaseio.com/",
    "storageBucket": "faceattendancerealtime-94252.appspot.com"
})


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(3,640)
cap.set(4,480)

imgBackground = cv2.imread('Resources/background1.png')


#importing the images into the list
folderModePath = 'Resources/Modes'
#for the folder path
modePathList = os.listdir(folderModePath)
imgModeList = []

for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))


#load the encoding file
print("Loading Encode file....")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()

encodeListKnown, studentIds = encodeListKnownWithIds
print(studentIds)
print("Encode file loaded...")

modeType = 0
counter = 0
id = -1
bucket = storage.bucket()
imgStudent = []

with open('attendance.csv', mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['ID', 'Name', 'Total Attendance'])


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:

    success, img = cap.read()

    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(imgS, scaleFactor=1.3, minNeighbors=5)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[40:40 + 633, 808:808 + 414] = imgModeList[modeType]

    faceThreshold = 0.5

    if True:
        for encodeFace,  faceloc in zip(encodeCurFrame,  faceCurFrame):

            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            # roi_gray = imgS[y:y + h, x:x + w]
            # roi_color = img[y:y + h, x:x + w]
            #
            # skin_color_lower = np.array([0, 20, 70], dtype=np.uint8)
            # skin_color_upper = np.array([20, 255, 255], dtype=np.uint8)
            # hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
            # mask = cv2.inRange(hsv, skin_color_lower, skin_color_upper)
            # skin_color_percentage = cv2.countNonZero(mask) / float(w * h)

            # print("Matches:", matches)
            print("Face Distance: " ,faceDis)
            matchIndex = np.argmin(faceDis)
            print("Matches: ", matches)
            print("Matches: ", matches[0], matches[1])
            print("Match Index: ", matchIndex)

        if matches[matchIndex] and faceDis[matchIndex] < faceThreshold:
            print("Known Face Detected")
            print(matches[matchIndex])
            print(studentIds[matchIndex])
            y1, x2, y2, x1 = faceloc
            # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
            id = studentIds[matchIndex]

            if counter == 0:
                cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                cv2.imshow("Face Attendance", imgBackground)
                cv2.waitKey(1)
                counter = 1
                modeType = 1

        else:
            cvzone.putTextRect(imgBackground, "Unknown Face", (275, 400))
            cv2.waitKey(1)
            continue


        if counter != 0:
            if counter == 1:
                #get data
                studentInfo = db.reference(f'Students/{id}').get()
                print(studentInfo)

                #get the image from the storage
                blob = bucket.get_blob(f'Images/{id}.png')
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

                #update data of attendance
                datetimeObject = datetime.strptime(studentInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")

                secondsElapsed = (datetime.now()-datetimeObject).total_seconds()
                print(secondsElapsed)

                if secondsElapsed > 30:
                    ref = db.reference(f'Students/{id}')
                    studentInfo['total_attendance'] += 1
                    #sending an update in database
                    ref.child('total_attendance').set(studentInfo['total_attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    with open('attendance.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([id, studentInfo['name'], studentInfo['last_attendance_time']])
                else:
                    modeType = 3
                    counter = 0
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

            if modeType != 3:
                if 10<counter<20:
                    modeType = 2

                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                if counter <= 10:
                    cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861,125), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
                    cv2.putText(imgBackground, str(studentInfo['course']), (1006, 550), cv2.FONT_HERSHEY_COMPLEX, 0.5,(255, 255, 255), 1)
                    cv2.putText(imgBackground, str(id), (1006, 493), cv2.FONT_HERSHEY_COMPLEX, 0.5,(255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625), cv2.FONT_HERSHEY_COMPLEX, 0.5,(100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6,(100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6,(100, 100, 100), 1)
                    (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (414 - w) // 2
                    cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (50, 50, 50), 1)

                    imgBackground[175:175+216,909:909+216] = imgStudent


                counter += 1

                if counter >= 20:
                    counter = 0
                    modeType = 0
                    studentInfo = []
                    imgStudent = []
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    else:
        modeType = 0
        counter = 0

    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)

    # import os
    # import pickle
    # import numpy as np
    # import cv2
    # import face_recognition
    # import cvzone
    # import firebase_admin
    # from firebase_admin import credentials
    # from firebase_admin import db
    # from firebase_admin import storage
    # from datetime import datetime
    # import csv
    #
    # cred = credentials.Certificate("serviceAccountKey.json")
    # firebase_admin.initialize_app(cred, {
    #     "databaseURL": "https://faceattendancerealtime-94252-default-rtdb.firebaseio.com/",
    #     "storageBucket": "faceattendancerealtime-94252.appspot.com"
    # })
    #
    # import cv2
    # import numpy as np
    #
    #
    # def perform_liveness_detection(roi):
    #     # Calculate the variance of pixel intensities in the ROI
    #     variance = cv2.meanStdDev(roi)[1] ** 2
    #
    #     # Set a threshold for variance to differentiate between live and static faces
    #     threshold = 800
    #
    #     # If the variance is above the threshold, consider it as a live face, otherwise a static face
    #     if variance[0][0] > threshold:
    #         return False
    #     else:
    #         return True
    #
    #
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap.set(3, 640)
    # cap.set(4, 480)
    #
    # imgBackground = cv2.imread('Resources/background1.png')
    #
    # # importing the images into the list
    # folderModePath = 'Resources/Modes'
    # # for the folder path
    # modePathList = os.listdir(folderModePath)
    # imgModeList = []
    #
    # for path in modePathList:
    #     imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
    #
    # # load the encoding file
    # print("Loading Encode file....")
    # file = open('EncodeFile.p', 'rb')
    # encodeListKnownWithIds = pickle.load(file)
    # file.close()
    #
    # encodeListKnown, studentIds = encodeListKnownWithIds
    # print(studentIds)
    # print("Encode file loaded...")
    #
    # modeType = 0
    # counter = 0
    # id = -1
    # bucket = storage.bucket()
    # imgStudent = []
    #
    # with open('attendance.csv', mode='w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow(['ID', 'Name', 'Total Attendance'])
    #
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    #
    # while True:
    #
    #     success, img = cap.read()
    #
    #     imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    #     imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    #     faces = face_cascade.detectMultiScale(imgS, scaleFactor=1.1, minNeighbors=5)
    #
    #     faceCurFrame = face_recognition.face_locations(imgS)
    #     encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
    #
    #     imgBackground[162:162 + 480, 55:55 + 640] = img
    #     imgBackground[40:40 + 633, 808:808 + 414] = imgModeList[modeType]
    #
    #     faceThreshold = 0.5
    #     liveness_threshold = 0.3
    #
    #     if True:
    #         for encodeFace, (x, y, w, h) in zip(encodeCurFrame, faces):
    #
    #             matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
    #             faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
    #
    #             roi_gray = imgS[y:y + h, x:x + w]
    #             roi_color = img[y:y + h, x:x + w]
    #
    #             skin_color_lower = np.array([0, 20, 70], dtype=np.uint8)
    #             skin_color_upper = np.array([20, 255, 255], dtype=np.uint8)
    #             hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
    #             mask = cv2.inRange(hsv, skin_color_lower, skin_color_upper)
    #             skin_color_percentage = cv2.countNonZero(mask) / float(w * h)
    #
    #             # print("Matches:", matches)
    #             print("Face Distance: ", faceDis)
    #             matchIndex = np.argmin(faceDis)
    #             print("Matches: ", matches)
    #             print("Matches: ", matches[0], matches[1])
    #             print("Match Index: ", matchIndex)
    #             is_live = perform_liveness_detection(roi_color)
    #
    #             if matches[matchIndex] and faceDis[matchIndex] < faceThreshold and is_live == True:
    #                 # cv2.putText(img, 'Liveness: True', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    #                 print("Known Face Detected")
    #
    #                 print(matches[matchIndex])
    #                 print(studentIds[matchIndex])
    #                 y1, x2, y2, x1 = (x, y, w, h)
    #                 # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
    #                 bbox = y1 + 50, x2 + 100, y2, x1 + 70
    #                 imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
    #                 id = studentIds[matchIndex]
    #
    #                 if counter == 0:
    #                     cvzone.putTextRect(imgBackground, "Loading", (275, 400))
    #                     cv2.imshow("Face Attendance", imgBackground)
    #                     cv2.waitKey(1)
    #                     counter = 1
    #                     modeType = 1
    #
    #             else:
    #                 cvzone.putTextRect(imgBackground, "Unknown Face", (155, 400))
    #                 print("Fake face")
    #
    #         if counter != 0:
    #             if counter == 1:
    #                 # get data
    #                 studentInfo = db.reference(f'Students/{id}').get()
    #                 print(studentInfo)
    #
    #                 # get the image from the storage
    #                 blob = bucket.get_blob(f'Images/{id}.png')
    #                 array = np.frombuffer(blob.download_as_string(), np.uint8)
    #                 imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
    #
    #                 # update data of attendance
    #                 datetimeObject = datetime.strptime(studentInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
    #
    #                 secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
    #                 print(secondsElapsed)
    #
    #                 if secondsElapsed > 30:
    #                     ref = db.reference(f'Students/{id}')
    #                     studentInfo['total_attendance'] += 1
    #                     # sending an update in database
    #                     ref.child('total_attendance').set(studentInfo['total_attendance'])
    #                     ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    #                     with open('attendance.csv', 'a') as f:
    #                         writer = csv.writer(f)
    #                         writer.writerow([id, studentInfo['name'], studentInfo['last_attendance_time']])
    #                 else:
    #                     modeType = 3
    #                     counter = 0
    #                     imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    #
    #             if modeType != 3:
    #                 if 10 < counter < 20:
    #                     modeType = 2
    #
    #                 imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    #
    #                 if counter <= 10:
    #                     cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
    #                                 cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    #                     cv2.putText(imgBackground, str(studentInfo['course']), (1006, 550), cv2.FONT_HERSHEY_COMPLEX,
    #                                 0.5, (255, 255, 255), 1)
    #                     cv2.putText(imgBackground, str(id), (1006, 493), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255),
    #                                 1)
    #                     cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625), cv2.FONT_HERSHEY_COMPLEX,
    #                                 0.5, (100, 100, 100), 1)
    #                     cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6,
    #                                 (100, 100, 100), 1)
    #                     cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625),
    #                                 cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
    #                     (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
    #                     offset = (414 - w) // 2
    #                     cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
    #                                 cv2.FONT_HERSHEY_COMPLEX, 1,
    #                                 (50, 50, 50), 1)
    #
    #                     imgBackground[175:175 + 216, 909:909 + 216] = imgStudent
    #
    #                 counter += 1
    #
    #                 if counter >= 20:
    #                     counter = 0
    #                     modeType = 0
    #                     studentInfo = []
    #                     imgStudent = []
    #                     imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    #     else:
    #         modeType = 0
    #         counter = 0
    #
    #     cv2.imshow("Face Attendance", imgBackground)
    #     cv2.waitKey(1)