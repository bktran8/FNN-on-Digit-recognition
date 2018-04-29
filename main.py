import numpy as np
import scipy
import matplotlib.pyplot as plt
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.utils import to_categorical

from util import func_confusion_matrix

# load (downloaded if needed) the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# transform each image from 28 by28 to a 784 pixel vector
pixel_count = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], pixel_count).astype('float32')
x_test = x_test.reshape(x_test.shape[0], pixel_count).astype('float32')

# normalize inputs from gray scale of 0-255 to values between 0-1
x_train = x_train / 255
x_test = x_test / 255

# Please write your own codes in responses to the homework assignment 5

x_valid = x_train[50000:60000]
x_train = x_train[:50000]
y_valid = y_train[50000:60000]
y_train = y_train[:50000]

test = np.unique(y_test)
classes = np.unique(y_train)
n_test = len(test)
n_classes = len(classes)

# First Model
first_model = Sequential()
first_model.add(Dense(512, activation='relu', input_shape=(pixel_count,)))
first_model.add(Dense(512, activation='relu'))
first_model.add(Dense(n_classes, activation='sigmoid'))
first_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

training = first_model.fit(x_train, to_categorical(y_train),
                           batch_size=256, epochs=3, verbose=1, validation_data=(x_valid, to_categorical(y_valid)))
predict_first_model = first_model.predict(x_valid)
first_model_predictions = [np.argmax(predictions) for predictions in predict_first_model]


# Second Model
second_model = Sequential()
second_model.add(Dense(512, activation='relu', input_shape=(pixel_count,)))
second_model.add(Dense(512, activation='relu'))
second_model.add(Dense(n_classes, activation='sigmoid'))
second_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

second_training = second_model.fit(x_train, to_categorical(y_train),
                                   batch_size=256, epochs=3, verbose=1, validation_data=(x_valid, to_categorical(y_valid)))
predict_second_model = second_model.predict(x_valid)
second_model_predictions = [np.argmax(predictions) for predictions in predict_second_model]


# Third Model
third_model = Sequential()
third_model.add(Dense(512, activation='relu', input_shape=(pixel_count,)))
third_model.add(Dense(n_classes, activation='sigmoid'))
third_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

third_training = third_model.fit(x_train, to_categorical(y_train),
                                 batch_size=256, epochs=3, verbose=1, validation_data=(x_valid, to_categorical(y_valid)))
predict_third_model = third_model.predict(x_valid)
third_model_predictions = [np.argmax(predictions) for predictions in predict_second_model]


print("Model 1")
print(first_model_predictions[0], first_model_predictions[1], first_model_predictions[2], first_model_predictions[3])

cm_1, acc_1, recall_1, prediction_1 = func_confusion_matrix(y_valid, first_model_predictions)
print("cm: ", cm_1, ",\n acc: ", acc_1, ",\n recall: ", recall_1, "\n, prediction: ", prediction_1)

print("Model 2")
print(predict_second_model[0], predict_second_model[1], predict_second_model[2], predict_second_model[3])

cm_2, acc_2, recall_2, prediction_2 = func_confusion_matrix(y_valid, second_model_predictions)
print("cm: ", cm_2, "\n, acc: ", acc_2, "\n, recall: ", recall_2, "\n, prediction: ", prediction_2)

print("Model 3")
print(predict_third_model[0], predict_third_model[1], predict_third_model[2], predict_third_model[3])

cm_3, acc_3, recall_3, prediction_3 = func_confusion_matrix(y_valid, third_model_predictions)
print("cm: ", cm_3, "\n, acc: ", acc_3, "\n, recall: ", recall_3, "\n, prediction: ", prediction_3)


print("Best Model: {}".format(np.argmax([acc_1, acc_2, acc_3])+1))

incorrect = []
predict_labels = []
if (np.argmax([acc_1, acc_2, acc_3])+1) == 1:
    predict_labels = [np.argmax(i) for i in first_model.predict(x_test)]
    incorrect = [j for j in range(len(predict_labels)) if predict_labels[j] != y_test[j]]
elif (np.argmax([acc_1, acc_2, acc_3])+1) == 2:
    predict_labels = [np.argmax(i) for i in second_model.predict(x_test)]
    incorrect = [j for j in range(len(predict_labels)) if predict_labels[j] != y_test[j]]
elif (np.argmax([acc_1, acc_2, acc_3])+1) == 3:
    predict_labels = [np.argmax(i) for i in third_model.predict(x_test)]
    incorrect = [j for j in range(len(predict_labels)) if predict_labels[j] != y_test[j]]
print(incorrect)

def shape(position, idx):
    return position.imshow(x_test[incorrect[idx]].reshape(28, 28), cmap=plt.cm.gray)

def label(position, idx):
    return position.set_title(str(predict_labels[incorrect[idx]]) + " Correct: " + str(y_test[incorrect[idx]]))

row, col = plt.subplots(2, 5)

shape(col[0, 0], 0)
shape(col[0, 1], 1)
shape(col[0, 2], 2)
shape(col[0, 3], 3)
shape(col[0, 4], 4)
shape(col[1, 0], 5)
shape(col[1, 1], 6)
shape(col[1, 2], 7)
shape(col[1, 3], 8)
shape(col[1, 4], 9)

label(col[0, 0], 0)
label(col[0, 1], 1)
label(col[0, 2], 2)
label(col[0, 3], 3)
label(col[0, 4], 4)
label(col[1, 0], 5)
label(col[1, 1], 6)
label(col[1, 2], 7)
label(col[1, 3], 8)
label(col[1, 4], 9)


plt.show()
