import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from cifar10_model import CIFAR10DataLoader, CIFAR10Model

class ImagePredictor:
    def __init__(self, model_path, class_names):
        self.model = CIFAR10Model()
        self.model.load(model_path)
        self.class_names = class_names
    
    def preprocess_image(self, image_path):
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return img
    
    def predict(self, image):
        plt.imshow(image, cmap=plt.cm.binary)
        prediction = self.model.model.predict(np.array([image]) / 255.0)
        index = np.argmax(prediction)
        print(f'I think it is {self.class_names[index]}')
        plt.show()

if __name__ == "__main__":
    data_loader = CIFAR10DataLoader()
    predictor = ImagePredictor('image_classifier.keras', data_loader.class_names)
    image = predictor.preprocess_image('images/deer.jpg')
    predictor.predict(image)
