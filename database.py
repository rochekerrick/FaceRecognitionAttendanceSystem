import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://faceattendancerealtime-94252-default-rtdb.firebaseio.com/"
})

ref = db.reference('Students')

data = {
    "1221012":{
        "name": "Roche Kerrick Santiago",
        "course": "Computer Science",
        "starting_year": 2020,
        "total_attendance": 7,
        "standing": 'G',
        "year": 3,
        "last_attendance_time": "2023-03-13 00:54:34"
    },
    "852741":{
        "name": "Kier Emmanuelle Atendido",
        "course": "Computer Science",
        "starting_year": 2020,
        "total_attendance": 4,
        "standing": 'G',
        "year": 3,
        "last_attendance_time": "2023-03-13 00:54:44"
    },
    "963852":{
        "name": "Gabriel Frigillana",
        "course": "Computer Science",
        "starting_year": 2020,
        "total_attendance": 3,
        "standing": 'G',
        "year": 3,
        "last_attendance_time": "2023-03-13 00:54:14"
    },
    "12456":{
        "name": "Riellette Magalso",
        "course": "Computer Science",
        "starting_year": 2019,
        "total_attendance": 4,
        "standing": 'G',
        "year": 4,
        "last_attendance_time": "2023-03-13 00:54:34"
    }


}

for key, value in data.items():
    ref.child(key).set(value)