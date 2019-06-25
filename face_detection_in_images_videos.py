"""                                                                          *
*Author: Tabicho Brian                                                       *
*Date: 24/06/19                                                              *
*Aim: Multiple face detection in video and images using haar cascade features*
"""

import cv2


class FaceDetection:
    """This class calls on the haar_cascade_file which is used for face detection and also
       draws a rectangle around a face when a face is identified"""

    def __init__(self, haar_cascade_filepath):
        self.classifier = cv2.CascadeClassifier(haar_cascade_filepath)
        self.grey_image = None

    def detect_faces(self, frame):
        self.grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(self.grey_image, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)


class RecordVideo(FaceDetection):

    """This class turns on the camera at port 0 and continues capturing till closed.
       It also inherits from the FaceDetection class so it can be able to detect faces
       in the frame
    """

    def __init__(self, camera_port=0):
        self.cap = cv2.VideoCapture(camera_port)
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            FaceDetection('./cascade_files/haarcascade_frontalface_alt.xml').detect_faces(frame)

            cv2.imshow('Face Detector', frame)
            c = cv2.waitKey(1)
            if c == 27:
                break

    cv2.destroyAllWindows()


class ReadImage(FaceDetection):

    """
    This class reads an image from the specified filepath and uses the FaceDetection class to determine if
    there are any faces in the picture.
    """

    def __init__(self, path):
        self.img = cv2.imread(path, cv2.IMREAD_COLOR)

    def run(self):
        FaceDetection('./cascade_files/haarcascade_frontalface_alt.xml').detect_faces(self.img)
        cv2.imshow('Frame Detector', self.img)
        cv2.waitKey()


if __name__ == '__main__':
    RecordVideo().run()
    #ReadImage('images/spain_team.jpg').run()