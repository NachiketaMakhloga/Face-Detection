import cv2
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# classifier function helps us to classify the file we imported that xml file.
# frontal face means it will help in detecting face
video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    # for infinite loop because as soon as the face is detected it should not close the camera.
    ret, frame = video_capture.read()
 # now for turning on the camera it will be in the form of bgr we have to convert it the form of greyscale then only it will be able to cpature and check.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # converting bgr colour to grey cvt refers to convert.
   # now there will be a use of multiscale function for detecting cascade_faces
 # as soon as camera is open it should be able to read the captured video.
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=12,
        minSize=(30, 30),
    )
    # for loop for making up the rectangle x y width and height
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
        # x+w and y+h as they are together as horizontal and vertical and give some thickness to the rectangle
    cv2.imshow('Video', frame)
    # wait fuction is used as it will help to make it wait for sometime
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
