import sys
import cv2

from vaporwave import vaporize
from PIL import Image

class Vaporizor:

    def __init__(self, image_path):
        img = vaporize(image_path)

        cv2.namedWindow("pic", cv2.WINDOW_NORMAL)
        cv2.imshow("pic", img)
        cv2.imwrite("pic.png", img)
        cv2.destroyAllWindows()
        image = Image.open("pic.png")
        image.show()
        sys.exit()