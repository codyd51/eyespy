import cv2
import numpy as np
from typing import Optional, Tuple
from enum import Enum


class EyeSide(Enum):
    LEFT_EYE = 0
    RIGHT_EYE = 1


class Detector(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h

        self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        self.last_left_eyes = []
        self.last_right_eyes = []

    def process_frame(self, frame):
        # type: (np.ndarray) -> np.ndarray
        # convert to grayscale before running classifiers
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_classifier.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = self.eye_classifier.detectMultiScale(roi_gray)
            for bounding_box in eyes:
                roi_color = self.process_eye_bounding_box(roi_color, bounding_box)

            for eye_side in EyeSide:
                tracked_eye_bounds = self.get_tracked_eye(eye_side)
                if not tracked_eye_bounds:
                    continue
                print('Tracked {}: {}'.format(eye_side.name, tracked_eye_bounds))
                ex, ey, ew, eh = tracked_eye_bounds
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        return frame

    def process_eye_bounding_box(self, face_frame, bounding_box):
        # type: (np.ndarray, Tuple[int, int, int, int]) -> np.ndarray
        ex, ey, ew, eh = bounding_box

        # draw raw bounds
        raw_bounds_color = (0, 0, 255)
        cv2.rectangle(face_frame, (ex, ey), (ex + ew, ey + eh), raw_bounds_color, 2)

        # find center of eye rect
        eye_midx = ex + (ew / 2)
        face_midx = face_frame.shape[0] / 2
        # left side?
        if eye_midx < face_midx:
            eye_side = EyeSide.LEFT_EYE
        else:
            eye_side = EyeSide.RIGHT_EYE
        self.cache_eye_bounds(eye_side, (ex, ey, ew, eh))

        return face_frame

    def get_tracked_eye(self, eye_side):
        # type: (EyeSide) -> Tuple[int, int, int, int]
        if eye_side == EyeSide.LEFT_EYE:
            array = self.last_left_eyes
        else:
            array = self.last_right_eyes

        avg_x, avg_y, avg_w, avg_h = 0, 0, 0, 0
        if not len(array):
            return avg_x, avg_y, avg_w, avg_h

        for x, y, w, h in array:
            avg_x += x
            avg_y += y
            avg_w += w
            avg_h += h
        avg_x /= len(array)
        avg_y /= len(array)
        avg_w /= len(array)
        avg_h /= len(array)

        return int(avg_x), int(avg_y), int(avg_w), int(avg_h)

    def cache_eye_bounds(self, eye_side, bounds):
        # type: (EyeSide, Tuple[int, int, int, int]) -> None

        if eye_side == EyeSide.LEFT_EYE:
            array = self.last_left_eyes
        else:
            array = self.last_right_eyes
        array.append(bounds)
        if len(self.last_left_eyes) > 5:
            # pop the oldest frame
            array.pop(0)


class CameraStream(object):
    def __init__(self):
        self.last_frame = None # type: Optional[np.ndarray]

    def __enter__(self):
        self.capture = cv2.VideoCapture()
        self.capture.open(0)
        return self

    def get_frame(self):
        # type: () -> (Optional[np.ndarray])
        ret, frame = self.capture.read()
        if not ret:
            # use cached frame
            return self.last_frame
        self.last_frame = frame
        return frame

    def __exit__(self, exc_type, exc_val, exc_tb):
        cv2.destroyAllWindows()
        self.capture.release()
        return None

detector = Detector(1280, 780)
with CameraStream() as stream:
    while cv2.waitKey(30) == -1:
        # grab a camera frame
        frame = stream.get_frame()
        if frame is None:
            continue
        # process it
        frame = detector.process_frame(frame)
        cv2.imshow("Pupil detection!", frame)
