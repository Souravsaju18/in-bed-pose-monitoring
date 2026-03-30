import cv2

def preprocess_image(path):

    img = cv2.imread(path)

    if img is None:
        print("Error loading image:", path)
        return None

    img = cv2.resize(img, (640, 480))

    return img