import cv2
import math


def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def DetectFace(image, faceCascade):
    min_size = (20,20)
    image_scale = 2
    haar_scale = 1.1
    min_neighbors = 3
    haar_flags = 0

    print(image.shape)
    h, w, _ = image.shape

    #grayscale = cv2.CreateImage((image.width, image.height), 8, 1)
    #smallImage = cv2.resize(image,
    #                        (0, 0),
    #                        math.floor(w / image_scale),
    #                        math.floor(h / image_scale))
    #cv2.cvtColor(smallImage, smallImage, cv2.COLOR_RGB2GRAY)
    return image

    cv2.CvtColor(image, grayscale, cv2.CV_BGR2GRAY)
    cv2.Resize(grayscale, smallImage, cv2.CV_INTER_LINEAR)
    cv2.EqualizeHist(smallImage, smallImage)

    faces = cv2.HaarDetectObjects(
            smallImage, faceCascade, cv2.CreateMemStorage(0),
            haar_scale, min_neighbors, haar_flags, min_size)

    if faces:
        for ((x, y, w, h), n) in faces:
            pt1 = (int(x * image_scale), int(y * image_scale))
            pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
            cv2.Rectangle(image, pt1, pt2, cv2.RGB(255, 0, 0), 5, 8, 0)

    return image



class Detector(object):
    def __init__(self):
        pass

    def process_frame(self, frame):
        return frame


face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

capture = cv2.VideoCapture()
capture.open(0)

while cv2.waitKey(17) == -1 or True:
    ret, img = capture.read()
    if not ret:
        raise RuntimeError()

    # convert to grayscale before running classifiers
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow("Pupil detection!", img)

cv2.destroyAllWindows()
capture.release()
