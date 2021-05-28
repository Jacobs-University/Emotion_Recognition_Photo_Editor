import cv2
import datetime
import numpy as np
import os

from keras.preprocessing.image import img_to_array
from PIL import Image
from vaporizer import Vaporizor


class ImageProcessor:
    """
        author:             Parth Shukla,

        This class will provide certain image processing functionality, like:

            - adding emoji
            # doing vaporwave edits (yet to be added)
    """

    def __init__(self, caption, face_classifier, emotion_classifier):
        self.__caption = caption
        self.__face_classifier = face_classifier
        self.__emotion_classifier = emotion_classifier
        self.__emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.__video_capture = cv2.VideoCapture(0)
        self.__font = cv2.FONT_HERSHEY_SIMPLEX

    def execute(self):
        while True:
            _, frame = self.__video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.__face_classifier.detectMultiScale(gray)

            frame, text_X, text_Y = self.add_caption_to_image(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

            for (x, y, w, h) in faces:
                if w <= 150:
                    continue
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    prediction = self.__emotion_classifier.predict(roi)[0]
                    label = self.__emotion_labels[prediction.argmax()]
                    cv2.putText(frame, label, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'No Face Detected', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            cv2.imshow('Emotion Detector', frame)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                self.post_production(frame, text_X, text_Y, label)
                break

        self.__video_capture.release()
        cv2.destroyAllWindows()

    def add_caption_to_image(self, frame):
        text_size = cv2.getTextSize(self.__caption, self.__font, 1, 2)[0]
        text_X = int((self.__video_capture.get(3) - text_size[0]) / 2)
        text_Y = int((self.__video_capture.get(4) + text_size[1]) / 1.10)
        cv2.putText(frame, self.__caption, (text_X, text_Y), self.__font, 1, (0, 255, 255), 2)
        return frame, text_X, text_Y

    def post_production(self, frame, text_X, text_Y, label):
        img_name = (datetime.datetime.now()).strftime("%d_%m_%Y_%H_%M_%S")
        cv2.imwrite(img_name + ".png", frame)
        print("{} written!".format(img_name))
        img = cv2.imread(img_name + ".png")
        original = img.copy()
        img = np.array(img, dtype=np.float64)
        img[np.where(img > 255)] = 255
        img = np.array(img, dtype=np.uint8)
        cv2.putText(img, self.__caption, (text_X, text_Y), self.__font, 1, (0, 255, 255), 2)
        '''img = cv2.transform(img, np.matrix([[0, 0.5, 0],
                                            [0, 0.5, 0],
                                            [0, 0.25, 0]]))'''
        cv2.imwrite(img_name + "_edited.png", img)
        s_img = cv2.imread("emojis/" + label + ".png")
        l_img = cv2.imread(img_name + "_edited.png")
        x_offset = y_offset = 50
        l_img[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1]] = s_img

        image_name = img_name + "_added_emoji.png"
        cv2.imwrite(image_name, l_img)
        image = Image.open(image_name)
        image.show()
        self.deal_with_files([img_name + ".png", img_name + "_edited.png"])
        last_edit = input("Would you like to have a cooler edit? Enter 'y' if you do\n")
        if last_edit == 'y':
            Vaporizor(image_name)

    def deal_with_files(self, file_paths):
        for path in file_paths:
            os.remove(path)


