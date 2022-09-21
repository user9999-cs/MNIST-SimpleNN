import tensorflow as tf
from keras import Sequential
from keras.layers import Flatten, Dense


class CallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > 0.95:
            print("\nReached 95% accuracy. Stopping the training")
            self.model.stop_training = True


callBack = CallBack()
data = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation=tf.nn.relu),
    Dense(10, activation=tf.nn.softmax)
])


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(training_images, training_labels, epochs=50, callbacks=callBack)

model.evaluate(test_images, test_labels)

classfications = model.predict(test_images)
print(classfications[210])
print(test_labels[210])
