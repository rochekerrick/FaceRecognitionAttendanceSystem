import cv2

# Load the pre-trained face detection cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

def perform_liveness_detection(roi):
    # Calculate the variance of pixel intensities in the ROI
    variance = cv2.meanStdDev(roi)[1]**2

    # Set a threshold for variance to differentiate between live and static faces
    threshold = 800

    # If the variance is above the threshold, consider it as a live face, otherwise a static face
    if variance[0][0] > threshold:
        return False
    else:
        return True

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the region of interest (ROI) containing the detected face
        roi = gray[y:y + h, x:x + w]

        # Perform liveness detection on the ROI (you can implement your own liveness detection method here)
        is_live = perform_liveness_detection(roi)

        # Display the liveness detection result
        if is_live:
            cv2.putText(frame, 'Live', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Static', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame with detected faces and liveness detection result
    cv2.imshow('Face Liveness Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
