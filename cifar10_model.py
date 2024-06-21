import matplotlib.pyplot as plt
from keras import datasets, layers, models

class CIFAR10DataLoader:
    def __init__(self):
        self.class_names = (
            "Plane", "Car", "Bird", "Cat", "Deer",
            "Dog", "Frog", "Horse", "Ship", "Truck"
        )
    
    def load_data(self):
        (training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
        training_images, testing_images = training_images / 255.0, testing_images / 255.0
        return training_images, training_labels, testing_images, testing_labels
    
    def plot_sample_images(self, training_images, training_labels):
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(training_images[i], cmap=plt.cm.binary)
            plt.xlabel(self.class_names[training_labels[i][0]])
        plt.show()

class CIFAR10Model:
    def __init__(self):
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    def train(self, training_images, training_labels, testing_images, testing_labels):
        self.model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))
    
    def evaluate(self, testing_images, testing_labels):
        loss, accuracy = self.model.evaluate(testing_images, testing_labels)
        print(f'Loss: {loss}')
        print(f'Accuracy: {accuracy}')
    
    def save(self, file_path):
        self.model.save(file_path)
    
    def load(self, file_path):
        self.model = models.load_model(file_path)

if __name__ == "__main__":
    data_loader = CIFAR10DataLoader()
    training_images, training_labels, testing_images, testing_labels = data_loader.load_data()
    data_loader.plot_sample_images(training_images, training_labels)
    
    model = CIFAR10Model()
    model.train(training_images, training_labels, testing_images, testing_labels)
    model.evaluate(testing_images, testing_labels)
    model.save('image_classifier.keras')
