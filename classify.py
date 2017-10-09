# Mini Neural Network for classifying even and odd numbers
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np
import gc
from logger import Logger


def relabel(labels):
    for idx, item in enumerate(labels):
        if item % 2 == 0:
            labels[idx] = 0     # even number
        else:
            labels[idx] = 1     # odd number
    return labels


def get_data():
    (train_picture, test_picture), (train_label, test_label) = mnist.load_data()
    train_picture = train_picture.reshape(60000, 784).astype('float32')
    train_label = train_label.reshape(10000, 784).astype('float32')
    train_picture = train_picture / 255
    train_label = train_label / 255
    test_picture = relabel(test_picture)
    test_label = relabel(test_label)
    test_picture = to_categorical(test_picture, 2)
    test_label = to_categorical(test_label, 2)
    return train_picture, test_picture, train_label, test_label


def log_result(model, train_label, test_label):
    classes = ['even number', 'odd number']
    predicted_classes = []
    ground_truth = []
    predictions = model.predict(train_label)
    for item in test_label:
        ground_truth.append(np.argmax(item))
    for item in predictions:
        predicted_classes.append(np.argmax(item))
    print(model.summary())
    my_logger = Logger()
    my_logger.log_confusion(ground_truth, predicted_classes, classes)


def main():
    train_picture, test_picture, train_label, test_label = get_data()
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(784,)))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    model.fit(train_picture, test_picture, batch_size=64, epochs=10, validation_data=(train_label, test_label))
    model.evaluate(train_label, test_label, verbose=0)
    log_result(model, train_label, test_label)
    gc.collect()


if __name__ == "__main__":
    main()
