from keras.models import load_model
from audio_utils import AudioUtils
from image_processor import ImageProcessor
import cv2

def main():
    caption = AudioUtils.record_caption()
    image_processor = ImageProcessor(caption,
                   cv2.CascadeClassifier(
                   r'/Users/parth/PycharmProjects/Emotion_Detection_CNN/haarcascade_frontalface_default.xml'),
                   load_model(r'/Users/parth/PycharmProjects/Emotion_Detection_CNN/model.h5'))

    ImageProcessor.execute(image_processor)

main()